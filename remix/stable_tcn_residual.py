"""Finite-memory, zero-preserving gated TCN residual around a stable core."""
from __future__ import annotations
import torch


class ZeroGatedTCN(torch.nn.Module):
    def __init__(self,width=16,blocks=8):
        super().__init__();self.width=width;self.dilations=[2**i for i in range(blocks)]
        self.input=torch.nn.Conv1d(1,width,1,bias=False)
        self.convolutions=torch.nn.ModuleList(torch.nn.Conv1d(width,2*width,3,dilation=d,bias=False) for d in self.dilations)
        self.conditions=torch.nn.ModuleList(torch.nn.Linear(2,width) for _ in self.dilations)
        self.projections=torch.nn.ModuleList(torch.nn.Conv1d(width,width,1,bias=False) for _ in self.dilations)
        self.output=torch.nn.Conv1d(width,1,1,bias=False)
        torch.nn.init.zeros_(self.output.weight)

    def forward(self,dry,controls,state=None):
        condition=controls[:,None,:].expand(-1,dry.shape[1],-1) if controls.ndim==2 else controls
        value=self.input(dry[:,None,:]*21.4);states=[]
        for i,(d,conv,cond,projection) in enumerate(zip(self.dilations,self.convolutions,self.conditions,self.projections)):
            old=value.new_zeros(value.shape[0],self.width,2*d) if state is None else torch.cat(state[i],2).squeeze(0).reshape(value.shape[0],self.width,2*d)
            extended=torch.cat((old,value),2);raw=conv(extended)
            activation,gate=raw.split(self.width,1)
            update=torch.tanh(activation)*torch.sigmoid(gate+cond(condition).transpose(1,2))
            value=value+projection(update)/(len(self.dilations)**.5)
            history=extended[:,:,-2*d:].reshape(value.shape[0],-1)
            states.append((history[:,:self.width*d].unsqueeze(0),history[:,self.width*d:].unsqueeze(0)))
        return self.output(value).squeeze(1),tuple(states)


class StableTCNResidual(torch.nn.Module):
    def __init__(self,base,tcn):
        super().__init__();self.base=base;self.tcn=tcn;self.control_count=2
        self.layers=base.layers+len(tcn.dilations)
        self.export_state_widths=[r.hidden_size for r in base.rnn_layers]+[tcn.width*d for d in tcn.dilations]

    @property
    def state_floats_per_mono_stream(self):return sum(self.export_state_widths)*2

    def forward(self,dry,controls,state=None):
        if state is not None and len(state)!=self.layers:raise ValueError('wrong TCN residual state count')
        base,bs=self.base(dry,controls,None if state is None else state[:self.base.layers])
        residual,rs=self.tcn(dry,controls,None if state is None else state[self.base.layers:])
        output=base+residual
        if not torch.jit.is_tracing() and not torch.isfinite(output).all():raise ValueError('non-finite TCN output')
        return output,(*bs,*rs)
