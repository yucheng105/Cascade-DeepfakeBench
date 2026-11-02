"""Generic sequential controller; adapters own inference, controller owns decisions."""
from __future__ import annotations

import math
import time

from .metrics import final_classification_metrics
from .policy import validate_policy
from .video_scoring import route_video_scores, route_final_stage, UNCERTAIN


def run_cascade(records: list[dict], policy: dict, adapters: dict) -> dict:
    started=time.perf_counter()
    dataset_id=policy.get('dataset_id')
    if not isinstance(dataset_id,str):
        raise ValueError('Policy dataset_id required')
    policy=validate_policy(policy,dataset_id,allow_exploratory=True)
    identities={r['video_name']:r for r in records}
    if not records or len(identities)!=len(records) or any(
            not isinstance(r['video_name'],str) or not r['video_name'] or r['label'] not in (0,1,None)
            for r in records):
        raise ValueError('Runtime cohort requires unique identities and binary or unknown labels')
    labeled = [r['label'] is not None for r in records]
    if any(labeled) and not all(labeled):
        raise ValueError('Cohort must be fully labeled or fully unknown; do not mix evaluation roles')
    final={}
    trace=[]
    summaries=[]
    reached=list(records)
    # Pure boundary functions are shared with the existing two-stage implementation.
    for index,stage in enumerate(policy['stages']):
        names={r['video_name'] for r in reached}
        timing={}
        if reached:
            scores,timing=adapters[stage['detector']].infer(reached)
            scores=tuple(scores)
            if len(scores)!=len(names) or {s.video_name for s in scores}!=names:
                raise ValueError('Adapter output cohort must exactly match reached videos')
            for s in scores:
                if s.label!=identities[s.video_name]['label']:
                    raise ValueError('Adapter label changed')
                if not math.isfinite(s.score) or not 0<=s.score<=1 or s.frame_count<1:
                    raise ValueError('Invalid adapter score or frame count')
            routed=(route_final_stage(scores,stage['low']) if index==len(policy['stages'])-1
                    else route_video_scores(scores,stage['low'],stage['high']))
        else:
            routed=()
        uncertain=set()
        for s in routed:
            trace.append(dict(video_name=s.video_name,label=s.label,stage=index+1,
                              detector=stage['detector'],score=s.score,frames=s.frame_count,
                              decision=s.decision,low=stage['low'],high=stage['high'],
                              allocated_forward_ms=timing.get('per_video_forward_ms',{}).get(s.video_name)))
            if s.decision==UNCERTAIN:
                uncertain.add(s.video_name)
            else:
                if s.video_name in final:
                    raise ValueError('Video exited more than once')
                final[s.video_name]=(s,index+1)
        summaries.append(dict(stage=index+1,detector=stage['detector'],reached=len(names),
                              exited=len(routed)-len(uncertain),uncertain=len(uncertain),
                              timing=timing,model_loaded=bool(names)))
        reached=[r for r in reached if r['video_name'] in uncertain]
    if reached or set(final)!=set(identities):
        raise ValueError('Terminal stage must decide every original video')
    videos=[final[r['video_name']][0] for r in records]
    if all(labeled):
        metrics=final_classification_metrics(videos)
        metrics.update(tp=sum(v.label==1 and v.decision=='FAKE' for v in videos),
                       fn=sum(v.label==1 and v.decision=='REAL' for v in videos),
                       fp=sum(v.label==0 and v.decision=='FAKE' for v in videos),
                       tn=sum(v.label==0 and v.decision=='REAL' for v in videos))
    else:
        metrics={key: None for key in ('acc','far','frr','real_count','fake_count','tp','fn','fp','tn')}
        metrics['video_count']=len(videos)
    metrics.update(fnr=metrics['far'],fpr=metrics['frr'],
                   mean_detectors=sum(s['reached'] for s in summaries)/len(records))
    return dict(decisions=[dict(video_name=v.video_name,label=v.label,prediction=int(v.decision=='FAKE'),
                                decision=v.decision,exit_stage=final[v.video_name][1],score=v.score,
                                frame_count=v.frame_count) for v in videos],
                trace=trace,stages=summaries,metrics=metrics,
                controller_seconds=time.perf_counter()-started)
