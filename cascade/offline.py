"""Policy-driven, strict offline detector integration: python -m cascade.offline."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from .controller import run_cascade
from .model_loader import REPO_ROOT,sha256
from .offline_data import load_records
from .policy import DATASETS,DETECTORS,validate_policy

DEFAULT_CONFIGS={name:REPO_ROOT/'training/config/detector'/f'{name}.yaml' for name in DETECTORS}
DEFAULT_CONFIGS['sbi']=REPO_ROOT/'cascade/configs/sbi_unit_range.yaml'


def write_json(path: Path, value: object) -> None:
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def overrides(values: list[str]) -> dict[str,Path]:
    mapped={}
    for item in values:
        name,sep,value=item.partition('=')
        if not sep or not value or name not in DETECTORS or name in mapped:
            raise ValueError('Use unique known DETECTOR=PATH overrides')
        mapped[name]=Path(value).resolve()
    return mapped


def source_state() -> dict:
    def git(*args):
        return subprocess.run(['git','-C',str(REPO_ROOT),*args],capture_output=True,text=True,check=True).stdout.strip()
    try:
        return dict(commit=git('rev-parse','HEAD'),dirty=bool(git('status','--porcelain')),
                    # Captures exact runtime code even when verification precedes commit.
                    runtime_files={str(p.relative_to(REPO_ROOT)):sha256(p)
                                   for p in sorted((REPO_ROOT/'cascade').glob('*.py'))})
    except (OSError,subprocess.CalledProcessError):
        return dict(commit='unknown')


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--dataset',choices=tuple(DATASETS),default='FaceForensics++')
    parser.add_argument('--dataset-json',type=Path,required=True)
    parser.add_argument('--rgb-root',type=Path,required=True)
    parser.add_argument('--compression',choices=('raw','c23','c40'),default='c23')
    parser.add_argument('--max-frames',type=int,default=32)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--allow-exploratory',action='store_true')
    parser.add_argument('--checkpoint',action='append',default=[],metavar='DETECTOR=PATH')
    parser.add_argument('--config',action='append',default=[],metavar='DETECTOR=PATH')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    started=time.perf_counter()
    output_created=False
    try:
        if args.output.exists():
            raise ValueError('Output exists; choose a new run directory')
        if args.batch_size<1:
            raise ValueError('Batch size must be positive')
        raw_policy=json.loads(args.policy.read_text(encoding='utf-8-sig'))
        policy=validate_policy(raw_policy,DATASETS[args.dataset],allow_exploratory=args.allow_exploratory)
        records=load_records(args.dataset_json,args.rgb_root,args.dataset,
                             compression=args.compression,max_frames=args.max_frames)
        configs=overrides(args.config);checkpoints=overrides(args.checkpoint)
        names=[s['detector'] for s in policy['stages']]
        if (set(configs)|set(checkpoints))-set(names):
            raise ValueError('Overrides name detectors absent from policy')
        import torch
        from .adapters import FrameDetectorAdapter
        device=torch.device(args.device)
        if device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; choose --device cpu explicitly if needed')
        if device.type not in ('cpu','cuda'):
            raise ValueError('Only CPU and CUDA devices supported')
        torch.manual_seed(1024)
        torch.set_num_threads(4)
        if device.type=='cuda':torch.cuda.manual_seed_all(1024)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        defaults=dict(xception='xception_best.pth',ucf='ucf_best.pth',sbi='FFc23.tar')
        adapters={name:FrameDetectorAdapter(name,
                     configs.get(name,DEFAULT_CONFIGS[name]),
                     checkpoints.get(name,REPO_ROOT/'training/weights'/defaults[name]),
                     device,args.batch_size) for name in names}
        if not args.allow_exploratory:
            for name,adapter in adapters.items():
                for key in ('checkpoint_sha256','config_sha256'):
                    if policy[key][name]!=adapter.provenance[key]:
                        raise ValueError('Policy binding does not match runtime '+key+' for '+name)
        manifest=dict(schema_version=1,policy=policy,policy_sha256=sha256(args.policy),
                      dataset=args.dataset,compression=args.compression,
                      dataset_json=str(args.dataset_json.resolve()),dataset_json_sha256=sha256(args.dataset_json),
                      rgb_root=str(args.rgb_root.resolve()),sampling='first_N_numeric_sorted_frames',
                      max_frames=args.max_frames,batch_size=args.batch_size,
                      videos=[{**r,'frames':[dict(path=p,sha256=sha256(Path(p))) for p in r['frames']]} for r in records],
                      models={n:a.provenance for n,a in adapters.items()},source=source_state(),
                      environment=dict(python=platform.python_version(),torch=torch.__version__,
                                       cuda_runtime=torch.version.cuda,device=str(device),
                                       gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None),
                      allow_exploratory=args.allow_exploratory,
                      evaluation_status='functional_smoke_only' if args.allow_exploratory else 'bound_policy_execution')
        args.output.mkdir(parents=True,exist_ok=False)
        output_created=True
        write_json(args.output/'manifest.json',manifest)
        print(json.dumps(dict(status='running',device=str(device),videos=len(records),models=names)),flush=True)
        result=run_cascade(records,policy,adapters)
        from .metrics import operational_auc
        from .video_scoring import VideoScore
        for stage in result['stages']:
            scores=[VideoScore(r['video_name'],r['label'],r['score'],r['frames'])
                    for r in result['trace'] if r['stage']==stage['stage']]
            stage['auc_reached_cohort']=operational_auc(scores)
        result['policy_id']=policy['policy_id']
        result['evaluation_status']=manifest['evaluation_status']
        result['runtime_seconds']=time.perf_counter()-started
        result['timing_scope']=dict(forward='CUDA-synchronized model calls, excludes warmup/decode/H2D/model load',
                                   allocated_forward_ms='batch forward ms divided by frames, accumulated per video',
                                   stage='model load + warmup + decode + H2D + forward + aggregation + release',
                                   runtime='CLI setup/validation/hashing + controller, excludes output writing')
        write_json(args.output/'report.json',result)
        write_csv(args.output/'decisions.csv',list(result['decisions'][0]),result['decisions'])
        write_csv(args.output/'trace.csv',list(result['trace'][0]),result['trace'])
        # Full IDs, never ambiguous filename stems; reached cohorts are explicitly tagged.
        stage_rows=[dict(dataset=args.dataset,sample_id=r['video_name'],label=r['label'],
                         detector_name=r['detector'],fake_score=r['score'],score_type='probability',
                         inference_time_ms=r['allocated_forward_ms'],frame_count=r['frames'],
                         policy_id=policy['policy_id'],stage=r['stage'],status='ok') for r in result['trace']]
        write_csv(args.output/'reached_scores.csv',list(stage_rows[0]),stage_rows)
        print(json.dumps(dict(status='complete',output=str(args.output.resolve()),metrics=result['metrics'],
                              reached=[s['reached'] for s in result['stages']],seconds=result['runtime_seconds'])),flush=True)
        return 0
    except Exception as error:
        if output_created:
            write_json(args.output/'error.json',dict(status='failed',error_type=type(error).__name__,error=str(error)))
        parser.exit(2,f'error: {type(error).__name__}: {error}\n')


if __name__=='__main__':sys.exit(main())
