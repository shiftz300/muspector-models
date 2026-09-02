"""A zero-preserving readout with access to every frozen recurrent layer."""
from __future__ import annotations
import torch


def encode_skip(base,dry,controls,state=None):
    condition=base._official_controls(controls,dry.shape[1])
    value=dry.unsqueeze(-1)*base.input_coef;features=[value]
    previous=(None,)*base.layers if state is None else state;states=[]
    if not torch.jit.is_tracing():
        if dry.ndim!=2 or len(previous)!=base.layers or controls.shape[0]!=dry.shape[0]:raise ValueError('invalid skip-readout geometry')
        if not torch.isfinite(dry).all() or not torch.isfinite(controls).all() or torch.any(controls<0) or torch.any(controls>1):raise ValueError('invalid skip-readout input')
    for layer,old in zip(base.rnn_layers,previous):
        value,current=layer(torch.cat((value,condition),2),old);features.append(value);states.append(current)
    return torch.cat(features,2),tuple(states)


class StableSkipReadout(torch.nn.Module):
    def __init__(self,base,readout):
        super().__init__();self.base=base;self.readout=readout
        self.control_count=base.control_count;self.layers=base.layers
        self.export_state_widths=[layer.hidden_size for layer in base.rnn_layers]

    @property
    def state_floats_per_mono_stream(self):return self.base.state_floats_per_mono_stream

    def forward(self,dry,controls,state=None):
        hidden,state=encode_skip(self.base,dry,controls,state)
        return self.base.output_layer(hidden[:,:,-self.base.hidden_size:]).squeeze(-1)+self.readout(hidden,controls),state
