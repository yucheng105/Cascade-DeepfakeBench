"""Versioned execution evidence views; never execute a detector or alter routing."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import time


class NoFaceEvidence(ValueError):
    execution_outcome='unable_to_decide'


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def route_reason(score: float, stage: dict, final: bool) -> str:
    if final:
        return 'final_fake_ge_threshold' if score>=stage['low'] else 'final_real_lt_threshold'
    return 'real_le_low' if score<=stage['low'] else 'fake_ge_high' if score>=stage['high'] else 'continue_between'


def context_for(policy: dict, record: dict, *, unit: str, profiles=None, models=None, **metadata) -> dict:
    frames=record.get('source_frames') or []
    coverage={name:{k:v for k,v in p.items() if k in ('state','valid_face_frames','valid_frame_ids','face_count','planned_samples')}
              for name,p in (profiles or {}).items()}
    return dict(unit=unit, sample_id=record.get('video_name',record.get('window_id')),
                source_id=record.get('source_id'),session_id=record.get('session_id'),window_id=record.get('window_id'),
                start_seconds=record.get('start_seconds'),end_seconds=record.get('end_seconds'),
                observed_end_seconds=record.get('observed_end_seconds'),
                timestamp_basis=sorted({f['timestamp_basis'] for f in frames if 'timestamp_basis' in f}) or None,
                source_frames_sha256=canonical_hash(frames) if frames else None,
                source_frame_count=len(frames) if record.get('source_frames') is not None else None,profile_coverage=coverage,
                model_bindings={n:{k:v for k,v in m.items() if k in ('checkpoint_sha256','config_sha256','preprocessing_sha256')}
                                for n,m in (models or {}).items()},
                policy_id=policy.get('policy_id'),policy_sha256=canonical_hash(policy),
                provenance_status='available_fields_only; null/missing means unrecorded',**metadata)


def explain(policy: dict, trace: list, events: list, context: dict, *, outcome: str,
            reason=None, operational_failure=None) -> dict:
    sample=context.get('sample_id')
    selected=[e for e in events if not e.get('sample_ids') or sample in e['sample_ids']]
    failures=[e for e in selected if e['status'] in ('failed','unable_to_decide')]
    failure=failures[-1] if failures else None
    scored={r['stage']:r for r in trace if sample is None or r.get('video_name',sample)==sample}
    exit_stage=next((i for i,r in sorted(scored.items()) if r['decision'] in ('REAL','FAKE')),None)
    stages=[]
    for index,stage in enumerate(policy['stages'],1):
        r=scored.get(index)
        event=next((e for e in reversed(selected) if e['stage']==index),None)
        status='not_started';why=None
        if r is not None:status='scored'
        elif failure and index==failure['stage']:status=failure['status'];why=failure.get('reason')
        elif exit_stage is not None and index>exit_stage:status='not_executed';why='prior_exit'
        elif failure and index>failure['stage']:status='not_executed';why='upstream_failure'
        elif outcome in ('skipped','cancelled'):status='not_executed';why=reason or outcome
        elif event:status=event['status'];why=event.get('reason')
        elif outcome in ('failed','unable_to_decide','processing'):status='unreported';why='stage_location_unavailable'
        elif outcome=='decided':status='unreported';why='execution_evidence_unavailable'
        stages.append(dict(stage=index,detector=stage['detector'],execution_status=status,
            score=r['score'] if r else None,decision=r['decision'] if r else None,
            low=stage['low'],high=stage['high'],
            route_reason=route_reason(r['score'],stage,index==len(policy['stages'])) if r else None,
            reason=why,forward_evidence='valid_score_returned' if r else 'not_established',
            evidence_origin='recorded_score' if r else 'execution_event' if event else 'derived_disposition'))
    return dict(schema_version=1,context=context,observation_outcome=outcome,reason=reason,
                failure_stage=failure['stage'] if failure else None,
                failure_phase=failure.get('phase') if failure else None,
                operational_failure=operational_failure,stages=stages,
                score_is_calibrated_confidence=False,
                cost_scope='allocated forward is serial batch attribution, not individual wall latency')


def write_explanation_csv(path: Path, explanations: list) -> None:
    columns=['trace_schema_version','unit','sample_id','window_id','policy_sha256','stage','detector',
             'execution_status','route_reason','score','decision','low','high','reason','context_json']
    temporary=path.with_suffix('.tmp')
    with temporary.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        for x in explanations:
            context=x['context']
            for stage in x['stages']:
                writer.writerow(dict(stage,trace_schema_version=x['schema_version'],
                    **{k:context.get(k) for k in ('unit','sample_id','window_id','policy_sha256')},
                    context_json=json.dumps(context,ensure_ascii=False,sort_keys=True)))
    for attempt in range(20):
        try:os.replace(temporary,path);break
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)


def annotate_windows(report: dict) -> None:
    """Also used by restart recovery; completed immutable views are reused."""
    if report.get('trace_schema_version')!=1:return
    terminal={'decided','failed','unable_to_decide','skipped','cancelled'}
    operational=report.get('operational_failure')
    for row in report['windows']:
        previous=row.get('explanation')
        if (row['outcome'] in terminal and previous and previous['observation_outcome']==row['outcome']
                and previous['operational_failure']==operational):continue
        evidence=row.get('evidence',row)
        context=context_for(report['policy'],row,unit='window',profiles=evidence.get('profiles',row.get('profiles',{})),
            models=report.get('provenance',{}).get('models'),input_contract=report['input_contract'],rule_version='temporal_evidence_v1')
        context['policy_sha256']=report['policy_sha256']
        context['sample_id']='window_'+str(row['index'])
        row['explanation']=explain(report['policy'],evidence.get('trace',row.get('trace',[])),
            evidence.get('execution_events',row.get('execution_events',[])),context,
            outcome=row['outcome'],reason=row.get('reason'),operational_failure=operational)
