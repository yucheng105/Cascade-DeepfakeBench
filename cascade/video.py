"""Single uploaded MP4 -> auditable, unlabeled offline Cascade result."""
import argparse
import json
import os
from pathlib import Path
import platform
import time

from .controller import run_cascade
from .download_face_model import MODEL_PATH, MODEL_URL, MODEL_SHA256
from .model_loader import REPO_ROOT, sha256
from .offline import DEFAULT_CONFIGS, source_state, write_csv, write_json
from .policy import validate_policy
from .video_input import FACE_PROFILE, SAMPLING, VideoLimits, YuNetFaces, prepare_video

POLICIES = {name: REPO_ROOT/'cascade/policies'/f'uploaded_video_{name}.json'
            for name in ('early', 'all_stages')}


def status_file(output: Path, phase: str, message: str):
    temporary = output/'status.tmp'
    temporary.write_text(json.dumps(dict(phase=phase, message=message), ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, output/'status.json')


def run_video(video: Path, output: Path, *, policy_path: Path | None = None,
              device: str = 'cuda', max_frames: int = 32, batch_size: int = 16,
              face_model: Path = MODEL_PATH, allow_exploratory: bool = False, progress=None,
              original_filename: str | None = None):
    started = time.perf_counter()
    if output.exists():
        raise ValueError('輸出目錄已存在；請使用新的 run 名稱')
    if not allow_exploratory:
        raise ValueError('此入口使用探索性政策，請明確啟用 --allow-exploratory')
    policy_path = policy_path or POLICIES['early']
    policy = validate_policy(json.loads(policy_path.read_text(encoding='utf-8-sig')),
                             'uploaded_video', allow_exploratory=True)
    if policy.get('deployment_ready') is not False:
        raise ValueError('Demo 政策必須明確標記 deployment_ready=false')
    if video.suffix.lower() != '.mp4' or not video.is_file():
        raise ValueError('請選擇可讀取的 MP4')
    if not 0 < video.stat().st_size <= VideoLimits().max_bytes:
        raise ValueError('影片為空或超過 200 MiB 上限')
    if batch_size < 1 or not 1 <= max_frames <= 32:
        raise ValueError('batch_size 必須為正數；max_frames 需介於 1–32')
    output.mkdir(parents=True, exist_ok=False)

    def update(phase, message):
        status_file(output, phase, message)
        if progress:
            progress(phase, message)

    try:
        update('validating', '檢查環境與模型')
        import torch
        from .adapters import FrameDetectorAdapter
        target = torch.device(device)
        if target.type not in ('cpu', 'cuda') or (target.type == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('指定的裝置無法使用；如需 CPU 請明確選擇 --device cpu')
        torch.manual_seed(1024)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if target.type == 'cuda':
            torch.cuda.manual_seed_all(1024)
        names = [s['detector'] for s in policy['stages']]
        defaults = dict(xception='xception_best.pth', ucf='ucf_best.pth', sbi='FFc23.tar')
        adapters = {name: FrameDetectorAdapter(name, DEFAULT_CONFIGS[name],
                    REPO_ROOT/'training/weights'/defaults[name], target, batch_size,
                    sampling=SAMPLING, input_profile=FACE_PROFILE) for name in names}
        input_hash = sha256(video)
        update('preprocessing', '影片解碼、抽幀與最大人臉對齊')
        frames, preprocessing = prepare_video(video, output, VideoLimits(max_frames=max_frames),
                                              YuNetFaces(face_model), update)
        write_json(output/'preprocessing.json', preprocessing)
        record = dict(video_name='upload_'+input_hash, label=None, frames=frames)
        source = source_state()
        source['demo_html_sha256'] = sha256(REPO_ROOT/'cascade/demo.html')
        source['training_files'] = {str(p.relative_to(REPO_ROOT)): sha256(p)
                                    for p in sorted((REPO_ROOT/'training').rglob('*.py'))}
        manifest = dict(schema_version=1, input_video=str(video.resolve()), input_sha256=input_hash,
                        original_filename=original_filename or video.name, dataset_id='uploaded_video',
                        ground_truth=None, inference_label_tensor_placeholder=0,
                        policy=policy, policy_sha256=sha256(policy_path),
                        face_model=dict(path=str(face_model.resolve()), sha256=MODEL_SHA256, url=MODEL_URL),
                        preprocessing=preprocessing, videos=[record],
                        models={n:a.provenance for n,a in adapters.items()}, source=source,
                        environment=dict(python=platform.python_version(), torch=torch.__version__,
                                         cuda_runtime=torch.version.cuda, device=str(target),
                                         gpu=torch.cuda.get_device_name(target) if target.type=='cuda' else None),
                        evaluation_status='unlabeled_exploratory_demo', deployment_ready=False)
        write_json(output/'manifest.json', manifest)

        class ReportingAdapter:
            def __init__(self, name, adapter):
                self.name, self.adapter = name, adapter

            def infer(self, records):
                update('inference', '推論：'+self.name.upper())
                return self.adapter.infer(records)

        result = run_cascade([record], policy,
                             {n:ReportingAdapter(n, a) for n,a in adapters.items()})
        result.update(policy_id=policy['policy_id'], evaluation_status='unlabeled_exploratory_demo',
                      original_filename=original_filename or video.name, input_sha256=input_hash,
                      preprocessing=preprocessing, runtime_seconds=time.perf_counter()-started,
                      warnings=['探索性門檻；fake score 不是校準後的正確率。',
                                '每個取樣影格只分析最大人臉；未做人員追蹤。'],
                      timing_scope=dict(runtime='validation + hashing + decode/face processing + model load/warmup/forward; excludes upload and output writing',
                                        forward='synchronized model calls only; excludes model load, warmup, decode and H2D'))
        if preprocessing['valid_face_frames'] < preprocessing['sampled_frame_count']:
            result['warnings'].append('部分抽樣影格沒有有效人臉；判定僅涵蓋成功處理的影格。')
        write_json(output/'report.json', result)
        write_csv(output/'decisions.csv', list(result['decisions'][0]), result['decisions'])
        write_csv(output/'trace.csv', list(result['trace'][0]), result['trace'])
        update('complete', '推論完成')
        return result
    except Exception as error:
        details = getattr(error, 'preprocessing', None)
        if details is not None and not (output/'preprocessing.json').exists():
            write_json(output/'preprocessing.json', details)
        write_json(output/'error.json', dict(status='failed', error_type=type(error).__name__, error=str(error)))
        update('failed', str(error))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=POLICIES['early'])
    parser.add_argument('--face-model', type=Path, default=MODEL_PATH)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-frames', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--allow-exploratory', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = run_video(args.video, args.output, policy_path=args.policy, face_model=args.face_model,
                           device=args.device, max_frames=args.max_frames, batch_size=args.batch_size,
                           allow_exploratory=args.allow_exploratory,
                           progress=lambda phase, message: print(message, flush=True))
        print(json.dumps(dict(decision=result['decisions'][0], output=str(args.output.resolve())), ensure_ascii=False))
        return 0
    except Exception as error:
        parser.exit(2, str(error)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
