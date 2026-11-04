"""Explicit offline all-detector export. Normal cascade requests never call this."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import threading
import time

from .adapters import FrameDetectorAdapter, bind_face_profile
from .model_loader import REPO_ROOT, sha256
from .offline import DEFAULT_CONFIGS, overrides, source_state
from .policy import DETECTORS, execution_contract
from .preprocessing import ProfileCache, preflight
from .video_input import VideoLimits


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _records(spec):
    if spec.get('dataset_id') not in ('ffpp','celeb_df_v2','dfdc','uploaded_video'):
        raise ValueError('Unknown cohort dataset_id')
    rows=spec.get('records')
    if not isinstance(rows,list) or not rows:
        raise ValueError('Nonempty cohort records required')
    seen=set()
    for r in rows:
        for key in ('sample_id','source_id','group_id','label_provenance'):
            if not isinstance(r.get(key),str) or not r[key].strip():
                raise ValueError('Cohort requires '+key)
        if r['sample_id'] in seen or (r.get('label') is not None and type(r['label']) is not int) or r.get('label') not in (0,1,None):
            raise ValueError('Duplicate identity or invalid label')
        if not isinstance(r.get('lineage',{}),dict):
            raise ValueError('lineage must be an object of declared identifiers')
        if 'window_label_scope' in r and r['window_label_scope']!='all_windows':
            raise ValueError('window_label_scope must be all_windows when explicitly supplied')
        seen.add(r['sample_id'])
    return rows


def timing_costs(timing):
    return dict(allocated_forward_ms=timing.get('allocated_forward_ms'),
        stage_ms=1000*timing.get('stage_seconds',0.),
        preprocessing_ms=1000*timing.get('prepare_seconds',0.),
        model_load_ms=1000*timing.get('model_load_seconds',0.),
        warmup_ms=1000*timing.get('warmup_seconds',0.))


def export_cohort(spec: dict, output: Path, *, mode='crops', unit='video', device='cpu',
                  detectors=DETECTORS, max_frames=32, configs=None, checkpoints=None,
                  window_seconds=4., sample_hz=2., face_model=None, stride_seconds=None, window_contract_version=1) -> dict:
    import torch
    rows=_records(spec)
    names=list(detectors)
    if not names or len(set(names))!=len(names) or any(n not in DETECTORS for n in names):
        raise ValueError('Unique known detectors required')
    if mode not in ('crops','official','legacy') or unit not in ('video','window') or (unit=='window' and mode=='crops'):
        raise ValueError('Window export requires an MP4 face profile')
    if mode!='crops' and spec['dataset_id']!='uploaded_video':
        raise ValueError('Raw MP4 execution domain is uploaded_video; source dataset belongs in lineage')
    actual=execution_contract(names,unit=unit,mode=mode,max_frames=max_frames,
                              window_seconds=window_seconds,sample_hz=sample_hz,
                              stride_seconds=stride_seconds,window_contract_version=window_contract_version)
    configs,checkpoints=configs or {},checkpoints or {}
    if (set(configs)|set(checkpoints))-set(names):raise ValueError('Unused detector override')
    if output.exists():raise ValueError('Output exists; choose a new artifact directory')
    defaults=dict(xception='xception_best.pth',ucf='ucf_best.pth',sbi='FFc23.tar')
    output.mkdir(parents=True,exist_ok=False)
    samples=[]; scores=[]; models={}
    began=time.perf_counter()
    if unit=='window':
        from .streaming import OfficialWindowEngine, VideoSource
        from .video import POLICIES
        p=json.loads(POLICIES['early'].read_text(encoding='utf-8'))
        p['stages']=[dict(detector=n,low=0.,high=1.) for n in names]
        p['stages'][-1].update(low=.5,high=.5)
        engine=OfficialWindowEngine(output,device,mode,face_model,512*1024*1024,policy=p,
            window_seconds=window_seconds,sample_hz=sample_hz,configs=configs,checkpoints=checkpoints,
            stride_seconds=stride_seconds,window_contract_version=window_contract_version)
        try:
            models=engine.provenance['models']
            for record in rows:
                source=VideoSource(Path(record['video']),stop_event=threading.Event())
                source_hash=sha256(Path(record['video']))
                try:
                    from .windowing import WindowTimeline, mp4_events, frame_evidence
                    def legacy_windows():
                        previous=-1
                        for i in range(math.ceil(source.info['duration_seconds']/window_seconds-1e-10)):
                            start=i*window_seconds;end=min(start+window_seconds,source.info['duration_seconds'])
                            admitted=[]
                            for j in range(math.ceil((end-start)*sample_hz-1e-10)):
                                index=min(int((start+j/sample_hz)*source.info['fps']),source.info['frame_count']-1)
                                if index==previous:continue
                                previous=index
                                admitted.append(dict(frame_index=index,timestamp_seconds=index/source.info['fps'],image=source.read(index)))
                            yield dict(index=i,start_seconds=start,end_seconds=end,samples=admitted)
                    def versioned_windows():
                        timeline=WindowTimeline(record['source_id'],record['sample_id'],window_seconds=window_seconds,
                            stride_seconds=actual['stride_seconds'],sample_hz=sample_hz)
                        for event in mp4_events(source,sample_hz):
                            yield from timeline.accept(event)
                    for window in (versioned_windows() if window_contract_version==2 else legacy_windows()):
                        i,start,end,admitted=(window[k] for k in ('index','start_seconds','end_seconds','samples'))
                        identity=record['sample_id']+':window:'+str(i)
                        sample={k:v for k,v in record.items() if k not in ('video','frames','window_labels')}
                        sample.update(source_label=record.get('label'),source_label_provenance=record['label_provenance'])
                        labels=record.get('window_labels',{})
                        if not isinstance(labels,dict):raise ValueError('window_labels must map window indices to ground truth')
                        count=math.ceil(source.info['duration_seconds']/actual['stride_seconds']-1e-10)
                        if any(not isinstance(k,str) or not k.isdigit() or str(int(k))!=k or not 0<=int(k)<count for k in labels):
                            raise ValueError('window_labels contains an unexpected window index')
                        if str(i) in labels:
                            truth=labels[str(i)]
                            if not isinstance(truth,dict) or (truth.get('label') is not None and type(truth.get('label')) is not int) or \
                                    truth.get('label') not in (0,1,None) or not isinstance(truth.get('label_provenance'),str) or not truth['label_provenance'].strip():
                                raise ValueError('Window ground truth requires a binary/unknown label and provenance')
                            sample.update(label=truth.get('label'),label_provenance=truth['label_provenance'])
                        elif record.get('window_label_scope')=='all_windows':
                            pass  # An explicit caller assertion, preserved as unverified lineage.
                        else:
                            sample.update(label=None,label_provenance='unknown_window_ground_truth; source label is not a window label')
                        sample.update(sample_id=identity,source_sha256=source_hash,start_seconds=start,end_seconds=end,
                            window_index=i,tail=end-start<window_seconds,timestamp_basis='frame_index_over_fps',
                            source_metadata=source.info,
                            source_frames=[dict(frame_index=s['frame_index'],timestamp_seconds=s['timestamp_seconds'],
                                raw_pixels_sha256=hashlib.sha256(s['image'].tobytes()).hexdigest()) for s in admitted])
                        if window_contract_version==2:
                            sample.update(window_contract_version=2,window_id=window['window_id'],tail=window['tail'],
                                observed_end_seconds=window['observed_end_seconds'],nominal_end_seconds=window['nominal_end_seconds'],
                                source_frames=[frame_evidence(s) for s in admitted])
                        samples.append(sample)
                        result=engine.infer(window,full_scores=True)
                        for score in result['score_rows']:
                            timing=score.pop('timing'); name=score['detector']
                            timing['allocated_forward_ms']=timing.get('per_video_forward_ms',{}).get('window_'+str(i))
                            scores.append(dict(score,sample_id=identity,costs=timing_costs(timing),
                                coverage=result['profiles'][engine.mapping[name]]))
                    source.finish()
                finally:source.close()
        finally:engine.close()
    else:
        definitions=preflight(mode,names,face_model) if mode!='crops' and face_model is not None else (
            preflight(mode,names) if mode!='crops' else {})
        adapters={n:FrameDetectorAdapter(n,Path(configs.get(n,DEFAULT_CONFIGS[n])),
            Path(checkpoints.get(n,REPO_ROOT/'training/weights'/defaults[n])),device,
            sampling=actual['sampling'],input_profile=actual['profiles'][n]) for n in names}
        for n,a in adapters.items():
            if mode!='crops':bind_face_profile(a,definitions[actual['profiles'][n]])
        models={n:a.provenance for n,a in adapters.items()}
        for i,record in enumerate(rows):
            sample={k:v for k,v in record.items() if k not in ('frames','video')}
            cache=None
            if mode=='crops':
                frames=sorted(record['frames'],key=lambda p:int(Path(p).stem))[:max_frames]
                if not frames:raise ValueError('Empty crop sample')
                if len({str(Path(p).resolve()) for p in frames})!=len(frames):
                    raise ValueError('Duplicate crop frame identities')
                hashes=[dict(path=str(Path(p).resolve()),sha256=sha256(Path(p))) for p in frames]
                sample.update(source_sha256=digest([r['sha256'] for r in hashes]),source_frames=hashes)
            else:
                sample['source_sha256']=sha256(Path(record['video']))
                try:
                    kwargs={} if face_model is None else dict(face_model=face_model)
                    cache=ProfileCache(Path(record['video']),output/('sample_'+str(i)),mode,names,device,
                        VideoLimits(max_frames=max_frames),180,512*1024*1024,**kwargs)
                    sample['source_frames']=cache.raw['samples']
                except Exception as error:
                    samples.append(sample)
                    scores.extend(dict(sample_id=record['sample_id'],detector=n,status='failed',score=None,frames=0,
                        reason=str(error),costs=timing_costs({}),coverage={}) for n in names)
                    continue
            samples.append(sample)
            for n,a in adapters.items():
                prepared_started=time.perf_counter()
                try:
                    r=dict(video_name=record['sample_id'],label=record.get('label'))
                    coverage={}
                    if cache is not None:
                        prepared=cache.get(n);groups=prepared['face_groups']
                        r.update(frames=[p for g in groups for p in g['faces']],face_groups=groups)
                        coverage=prepared
                    else:r['frames']=frames
                    preparation=time.perf_counter()-prepared_started
                    result,timing=a.infer([r])
                    if len(result)!=1 or result[0].video_name!=record['sample_id'] or result[0].label!=r['label'] or \
                            not math.isfinite(result[0].score) or not 0<=result[0].score<=1 or result[0].frame_count<1:
                        raise ValueError('Invalid detector output')
                    timing.update(prepare_seconds=preparation,allocated_forward_ms=timing.get('per_video_forward_ms',{}).get(record['sample_id']))
                    scores.append(dict(sample_id=record['sample_id'],detector=n,status='ok',score=result[0].score,
                        frames=result[0].frame_count,costs=timing_costs(timing),coverage=coverage))
                except Exception as error:
                    scores.append(dict(sample_id=record['sample_id'],detector=n,status='failed',score=None,frames=0,
                        reason=str(error),costs=timing_costs({}),coverage=cache.summary() if cache else {}))
    payload=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in scores).encode()
    (output/'scores.jsonl').write_bytes(payload)
    manifest=dict(schema_version=1,kind='complete_detector_scores',dataset_id=spec['dataset_id'],
        detectors=names,input_contract=actual,models=models,samples=samples,
        cohort_sha256=digest(spec),scores_sha256=hashlib.sha256(payload).hexdigest(),
        exporter=source_state(),lineage_status='declared_unverified',
        cost_scope='allocated_forward_ms',latency_model='sum_reached_forward_ms_serial',
        hardware=dict(python=platform.python_version(),torch=torch.__version__,device=str(device),
            gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None),
        coverage=dict(expected=len(samples)*len(names),ok=sum(r['status']=='ok' for r in scores),
            failed=sum(r['status']!='ok' for r in scores),eligibility='all_declared_samples_required_no_silent_exclusion'),
        export_seconds=time.perf_counter()-began)
    encoded=json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False).encode()
    (output/'manifest.json').write_bytes(encoded)
    (output/'manifest.sha256').write_text(hashlib.sha256(encoded).hexdigest(),encoding='ascii')
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('crops','official','legacy'),default='crops')
    parser.add_argument('--unit',choices=('video','window'),default='video')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--detectors',default=','.join(DETECTORS))
    parser.add_argument('--checkpoint',action='append',default=[])
    parser.add_argument('--config',action='append',default=[])
    parser.add_argument('--max-frames',type=int,default=32)
    parser.add_argument('--window-seconds',type=float,default=4.)
    parser.add_argument('--sample-hz',type=float,default=2.)
    parser.add_argument('--stride-seconds',type=float)
    parser.add_argument('--window-contract-version',type=int,choices=(1,2),default=1)
    args=parser.parse_args(argv)
    spec=json.loads(args.cohort.read_text(encoding='utf-8-sig'))
    # Resolve the cohort's declared paths relative to the cohort, never the shell cwd.
    for row in spec['records']:
        for key in ('video','frames'):
            if key not in row:continue
            if key=='video':row[key]=str((args.cohort.parent/row[key]).resolve())
            else:row[key]=[str((args.cohort.parent/p).resolve()) for p in row[key]]
    result=export_cohort(spec,args.output,mode=args.mode,unit=args.unit,device=args.device,
        detectors=tuple(args.detectors.split(',')),configs=overrides(args.config),
        checkpoints=overrides(args.checkpoint),max_frames=args.max_frames,window_seconds=args.window_seconds,
        sample_hz=args.sample_hz,stride_seconds=args.stride_seconds,window_contract_version=args.window_contract_version)
    print(json.dumps(result['coverage']))


if __name__=='__main__':main()
