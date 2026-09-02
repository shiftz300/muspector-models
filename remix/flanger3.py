"""Safe-subdomain generic Flanger inverse using a Wet-only comb-phase search."""
from __future__ import annotations
import math,random
from pathlib import Path
import numpy as np,torch
from scipy.signal import stft
from .chorus3 import ChorusPairsV3,restore_chorus

RATE_MIN,RATE_MAX=1.0,5.0
BASE_MIN_MS,BASE_MAX_MS=1.8,5.5
DEPTH_MIN_MS,DEPTH_MAX_MS=.35,1.4
MIX_MIN,MIX_MAX=.25,.45
ADMITTED_RATE_MAX=2.3
ADMITTED_MIX_MAX=.38
PHASE_CANDIDATES=256

def flanger(clean,rng):
 v={"family":"flanger","rate_hz":math.exp(rng.uniform(math.log(RATE_MIN),math.log(RATE_MAX))),"base_ms":rng.uniform(BASE_MIN_MS,BASE_MAX_MS),"depth_ms":rng.uniform(DEPTH_MIN_MS,DEPTH_MAX_MS),"mix":rng.uniform(MIX_MIN,MIX_MAX),"renderer":"muspector-fractional-delay-flanger-v1"};v["depth_ms"]=min(v["depth_ms"],v["base_ms"]-.5);phase=rng.uniform(0,2*np.pi);v["hidden_phase_radians"]=phase;t=np.arange(len(clean))/48000;lfo=np.sin(2*np.pi*v["rate_hz"]*t+phase);delay=(v["base_ms"]+v["depth_ms"]*lfo)*48;shift=np.interp(np.arange(len(clean))-delay,np.arange(len(clean)),clean,left=0,right=0);return ((1-v["mix"])*clean+v["mix"]*shift).astype("float32"),v,lfo.astype("float32")
def controls(v):
 return np.array([math.log(v["rate_hz"]/RATE_MIN)/math.log(RATE_MAX/RATE_MIN),(v["base_ms"]-BASE_MIN_MS)/(BASE_MAX_MS-BASE_MIN_MS),(v["depth_ms"]-DEPTH_MIN_MS)/(DEPTH_MAX_MS-DEPTH_MIN_MS),(v["mix"]-MIX_MIN)/(MIX_MAX-MIX_MIN),0],dtype="float32")
def admitted(v):return v["rate_hz"]<ADMITTED_RATE_MAX and v["mix"]<=ADMITTED_MIX_MAX
def estimate_phase(wet,v):
 N,H=4096,240;_,times,z=stft(wet,fs=48000,window="hann",nperseg=N,noverlap=N-H,nfft=N,boundary=None,padded=False);logmag=np.log(np.abs(z).T+1e-7);logmag-=logmag.mean(0,keepdims=True);cep=np.fft.irfft(logmag,n=N,axis=1);cep/=np.sqrt(np.mean(cep[:,20:400]**2,axis=1,keepdims=True)+1e-12);grid=np.linspace(0,2*np.pi,PHASE_CANDIDATES,endpoint=False);scores=[]
 for phase in grid:
  delay=(v["base_ms"]+v["depth_ms"]*np.sin(2*np.pi*v["rate_hz"]*times+phase))*48;lower=np.floor(delay).astype(int);fraction=delay-lower;ridge=cep[np.arange(len(times)),lower]*(1-fraction)+cep[np.arange(len(times)),lower+1]*fraction;scores.append(float(np.mean(ridge)))
 return float(grid[int(np.argmax(scores))])
def restore_flanger(wet,v):
 if not admitted(v):return np.asarray(wet,dtype="float32").copy(),False,np.zeros(len(wet),dtype="float32")
 phase=estimate_phase(wet,v);time=np.arange(len(wet))/48000;lfo=np.sin(2*np.pi*v["rate_hz"]*time+phase).astype("float32");return restore_chorus(wet,v,lfo),True,lfo
class FlangerPairsV3(torch.utils.data.Dataset):
 def __init__(self,workspace:Path,split:str,samples:int,target_frames:int,seed:int):
  self.base=ChorusPairsV3(workspace,split,samples,target_frames,seed);self.samples=samples;self.seed=seed;self.authorization=self.base.authorization
 def __len__(self):return self.samples
 def realized_source_counts(self):return self.base.realized_source_counts()
 def __getitem__(self,index):
  row=dict(self.base[index]);wet,v,lfo=flanger(row["clean"].numpy(),random.Random(self.seed+index*104729));row.update({"wet":torch.from_numpy(wet),"controls":torch.from_numpy(controls(v)),"lfo":torch.from_numpy(lfo),"control_values":v});return row
def manifest():
 return {"schema":1,"architecture":"wet-real-cepstrum-moving-comb-phase-search-plus-causal-fractional-delay-inverse","mechanism":"modulation","family":"flanger","parameters":0,"phase_candidates":PHASE_CANDIDATES,"admitted_rate_hz":[RATE_MIN,ADMITTED_RATE_MAX],"admitted_mix":[MIX_MIN,ADMITTED_MIX_MAX],"ambiguous_input_behavior":"abstain-and-pass-through","hidden_forward_state_is_inference_input":False,"graph_order_input":False,"neighbouring_effect_input":False,"clean_or_oracle_input":False,"physical_device_claim":False,"causal_inverse_after_noncausal_bounded_phase_estimation":True}
