"""Versioned evidence summaries; no score calibration or stage routing."""
from __future__ import annotations

import copy
import math
from collections import Counter

TERMINAL = frozenset(('decided', 'unable_to_decide', 'skipped', 'cancelled', 'failed'))


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def gaps(intervals, horizon):
    result = []
    previous = 0.
    for start, end in union(intervals):
        if start > previous:
            result.append([previous, start])
        previous = max(previous, end)
    if previous < horizon:
        result.append([previous, horizon])
    return result


def intersection(left, right):
    left,right=union(left),union(right)
    i=j=0
    result=[]
    while i<len(left) and j<len(right):
        start,end=max(left[i][0],right[j][0]),min(left[i][1],right[j][1])
        if start<end:result.append([start,end])
        if left[i][1]<right[j][1]:i+=1
        else:j+=1
    return result


class TemporalAggregator:
    """Immutable terminal evidence; timing updates do not revise decisions."""

    def __init__(self):
        self.results = {}
        self.final_reason = None

    def update(self, row):
        if row['outcome'] not in TERMINAL:
            return
        keys = ('window_id','index','start_seconds','end_seconds','observed_end_seconds',
                'outcome','decision','score','exit_stage','trace','profiles','source_frames','reason')
        item = {k:copy.deepcopy(row[k]) for k in keys if k in row}
        identity = row['window_id']
        start,end=row['start_seconds'],row.get('observed_end_seconds',row['end_seconds'])
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (start,end)) or start<0 or end<start:
            raise ValueError('Invalid WindowResult observation interval')
        if identity in self.results and self.results[identity] != item:
            raise ValueError('Conflicting terminal WindowResult: '+identity)
        if self.final_reason is not None and identity not in self.results:
            raise ValueError('Aggregator is finalized')
        if item.get('outcome') == 'decided':
            score = item.get('score')
            if end<=start or item.get('decision') not in ('REAL','FAKE') or isinstance(score,bool) or \
                    not isinstance(score,(int,float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid decided WindowResult')
        self.results[identity] = item

    def snapshot(self, horizon, *, pending=0, finalization_reason=None):
        if not math.isfinite(horizon) or horizon < 0:
            raise ValueError('Invalid observed horizon')
        if finalization_reason is not None:
            if finalization_reason not in ('eof','user_stop','failure'):
                raise ValueError('Invalid finalization reason')
            if self.final_reason not in (None, finalization_reason):
                raise ValueError('Conflicting finalization')
            self.final_reason = finalization_reason
        rows = [self.results[k] for k in sorted(self.results)]
        def interval(row):
            return [row['start_seconds'], min(horizon,row.get('observed_end_seconds',row['end_seconds']))]
        decided = [r for r in rows if r['outcome']=='decided']
        fake = [r for r in decided if r['decision']=='FAKE']
        real = [r for r in decided if r['decision']=='REAL']
        unknown = [r for r in rows if r['outcome']!='decided']
        covered = union([interval(r) for r in decided])
        missing = gaps(covered,horizon)
        state = ('FAKE_EVIDENCE' if fake else 'REAL_EVIDENCE_ONLY' if real and not unknown and
                 not missing and not pending and self.final_reason!='failure' else 'INSUFFICIENT_EVIDENCE')
        profiles = {}
        frames = set()
        for row in rows:
            index_to_id = {s['frame_index']:s['frame_id'] for s in row.get('source_frames',[])}
            frames.update(s['frame_id'] for s in row.get('source_frames',[]))
            for name, profile in row.get('profiles',{}).items():
                profiles.setdefault(name,set()).update(index_to_id.get(i,str(i)) for i in profile.get('valid_frame_ids',[]))
        return dict(rule_version='temporal_evidence_v1',state=state,provisional=self.final_reason is None,
                    finalized=self.final_reason is not None,finalization_reason=self.final_reason,
                    observed_horizon_seconds=horizon,source_window_ids=[r['window_id'] for r in rows],
                    fake_window_ids=[r['window_id'] for r in fake],
                    fake_intervals=union([interval(r) for r in fake]),real_intervals=union([interval(r) for r in real]),
                    conflicting_intervals=intersection([interval(r) for r in fake],[interval(r) for r in real]),
                    decided_intervals=covered,decided_seconds=sum(b-a for a,b in covered),
                    gap_intervals=missing,unknown_intervals=union([interval(r) for r in unknown]),
                    unknown_windows=[dict(window_id=r['window_id'],outcome=r['outcome'],
                        interval=interval(r),reason=r.get('reason')) for r in unknown],
                    dispositions=dict(sorted(Counter(r['outcome'] for r in rows).items())),pending=pending,
                    sampled_unique_frames=len(frames),valid_profile_frames={k:len(v) for k,v in sorted(profiles.items())},
                    interpretation='Observed window evidence only; interval coverage is not per-frame or per-person coverage')
