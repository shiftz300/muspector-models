"""Wet-only hidden fractional-delay trajectory estimator for Chorus."""

from __future__ import annotations

import torch
from torch import nn

from .chorus3 import CONTROL_WIDTH, RATE_MAX, RATE_MIN
from .foundation_data import RATE


class _Block(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__(); self.temporal=nn.Conv2d(channels,channels*2,(3,3),padding=(1,dilation),dilation=(1,dilation),groups=channels); self.mix=nn.Conv2d(channels,channels,1)
    def forward(self,x):
        a,b=self.temporal(x).chunk(2,1); return x+0.25*self.mix(torch.tanh(a)*torch.sigmoid(b))


class ChorusTrajectoryV3(nn.Module):
    mechanism="modulation"; family="chorus"
    def __init__(self, channels=16, depth=8, n_fft=1024, hop=240):
        super().__init__(); self.channels,self.depth,self.n_fft,self.hop=channels,depth,n_fft,hop; self.dilations=tuple(2**i for i in range(depth))
        self.register_buffer("window",torch.hann_window(n_fft),persistent=False)
        self.stem=nn.Conv2d(4+CONTROL_WIDTH,channels,(5,3),padding=(2,1)); self.blocks=nn.ModuleList(_Block(channels,d) for d in self.dilations)
        self.trajectory=nn.Conv2d(channels,1,1); self.uncertainty=nn.Conv2d(channels,1,1)
        nn.init.zeros_(self.trajectory.weight);nn.init.zeros_(self.trajectory.bias);nn.init.zeros_(self.uncertainty.weight);nn.init.constant_(self.uncertainty.bias,-3.)
    def forward(self,wet,controls):
        if wet.ndim!=2 or controls.shape!=(wet.shape[0],CONTROL_WIDTH): raise ValueError("invalid Chorus model input")
        s=torch.stft(wet,self.n_fft,self.hop,window=self.window.to(wet),center=True,pad_mode="constant",return_complex=True)
        scale=s.abs().mean((1,2),keepdim=True).clamp_min(1e-6); mag=torch.log1p(s.abs()/scale); delta=torch.nn.functional.pad(mag[:,:,1:]-mag[:,:,:-1],(1,0))
        features=torch.stack((mag,delta),1)[:,:,:512,:]
        features=torch.nn.functional.avg_pool2d(features,(16,1),(16,1))
        condition=controls.mul(2).sub(1)[:,:,None,None].expand(-1,-1,32,features.shape[-1])
        rate=RATE_MIN*torch.exp(controls[:,0]*torch.log(controls.new_tensor(RATE_MAX/RATE_MIN)))
        frame_time=torch.arange(features.shape[-1],device=wet.device,dtype=wet.dtype)*self.hop/RATE
        angle=2*torch.pi*rate[:,None]*frame_time[None,:]
        clocks=torch.stack((torch.sin(angle),torch.cos(angle)),1)[:,:,None,:].expand(-1,-1,32,-1)
        h=torch.tanh(self.stem(torch.cat((features,condition,clocks),1)))
        for block in self.blocks:h=block(h)
        trajectory=torch.tanh(self.trajectory(h).mean(2).squeeze(1)); uncertainty=torch.nn.functional.softplus(self.uncertainty(h).mean(2).squeeze(1))+1e-5
        return trajectory,uncertainty
    def manifest(self):
        return {"schema":1,"architecture":"wet-stft-comb-trajectory-tcn-with-known-rate-clock","mechanism":self.mechanism,"family":self.family,"channels":self.channels,"depth":self.depth,"n_fft":self.n_fft,"hop":self.hop,"parameters":sum(p.numel() for p in self.parameters()),"hidden_forward_state":"lfo_start_phase","hidden_forward_state_is_inference_input":False,"known_rate_clock_input":True,"graph_order_input":False,"neighbouring_effect_input":False,"clean_or_oracle_input":False,"causal":False,"uncertainty_output":True}
