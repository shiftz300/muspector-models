"""Bounded, audio-only recovery of rejected classifier proposals.

Every fit here is a diagnostic order-scoring model. Its gain, parallel-clean
fraction and alternative controls must never silently replace the actual
ChainSpec or be applied to source/output audio.
"""
from __future__ import annotations
import numpy as np
from .order_search import SEARCH_RENDERERS,convex_renderer_fit,decode_controls
from .render import render_chain
from .spec import ChainSpec,ranked_topologies

MENUS=('parallel-clean','bounded-knobs','combined')
CLEAN_FRACTIONS=(0.,.1,.25)
CONTROL_RADIUS=.1


def raw_proposal(row,selection):
    name,weight,margin=selection
    p=np.clip(np.asarray(row['probabilities'][name]),1e-6,1-1e-6)
    logits=(1-weight)*np.asarray(row['logits'])+weight*np.log(p/(1-p))
    ranked=ranked_topologies(row['active'],logits)
    if len(ranked)>1 and ranked[0][1]-ranked[1][1]<margin:return tuple(row['baseline'])
    return tuple(ranked[0][0])


def needs_fit(row,selection):
    if len(row['active'])<2:return False
    candidate=raw_proposal(row,selection)
    error=next(v['error'] for v in row['errors'] if tuple(v['topology'])==candidate)
    return candidate!=tuple(row['baseline']) and error>row['baseline_error']+1e-12


def control_variants(controls,active):
    initial=np.asarray(controls,dtype=np.float64)
    if initial.shape!=(9,) or not np.isfinite(initial).all() or np.any((initial<0)|(initial>1)):
        raise ValueError('nine finite normalized controls required')
    variants=[('base',initial.copy())];seen={tuple(initial)}
    for family,index in (('drive',0),('delay',5),('reverb',8)):
        if family not in active:continue
        for sign in (-1,1):
            value=initial.copy();value[index]=np.clip(value[index]+sign*CONTROL_RADIUS,0,1)
            if tuple(value) not in seen:
                variants.append((f'{family}-{sign:+d}',value));seen.add(tuple(value))
    return variants


def fit_topology_options(dry,wet,topology,controls,sample_rate=44100,menus=MENUS):
    x=np.asarray(dry);y=np.asarray(wet)
    if x.ndim!=1 or x.shape!=y.shape or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('equal finite mono diagnostic pairs required')
    if not menus or any(name not in MENUS for name in menus):raise ValueError('unknown bank menu')
    best={name:None for name in menus}
    for tag,variant in control_variants(controls,topology):
        if tuple(menus)==('parallel-clean',) and tag!='base':continue
        spec=ChainSpec(decode_controls(topology,variant))
        rendered=[render_chain(x,spec,sample_rate,renderer) for renderer in SEARCH_RENDERERS]
        for clean in CLEAN_FRACTIONS:
            if tuple(menus)==('bounded-knobs',) and clean!=0:continue
            candidates=[(1-clean)*value+clean*x for value in rendered]
            error,weights=convex_renderer_fit(candidates,y)
            item={'error':error,'fit_topology':list(topology),'normalized_controls':variant.tolist(),'variant':tag,
                  'parallel_clean_fraction':clean,'renderer_weights':weights,
                  'scoring_only':True,'gain_applied_to_output':False}
            applicable=['combined']
            if tag=='base':applicable.append('parallel-clean')
            if clean==0:applicable.append('bounded-knobs')
            for menu in applicable:
                if menu in best and (best[menu] is None or error<best[menu]['error']):best[menu]=item
    return best


def apply_bank_fit(row,selection,menu):
    """Return a new diagnostic row; all frozen controls and source rows stay put."""
    if menu not in MENUS:raise ValueError('unknown bounded diagnostic bank')
    result=dict(row);result['errors']=[dict(v) for v in row['errors']]
    if row.get('bank_fit'):
        candidate=raw_proposal(row,selection);fit=row['bank_fit'][menu]
        if tuple(fit.get('fit_topology',()))!=candidate:
            raise ValueError('diagnostic fit belongs to a different topology')
        for value in result['errors']:
            if tuple(value['topology'])==candidate:
                if fit['error']<value['error']:
                    value['error']=fit['error'];value['diagnostic_fit']=fit
    return result
