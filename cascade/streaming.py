"""Paced MP4 sessions with isolated resident inference and bounded pending work.

Only already admitted source samples cross into the inference worker. Native
decode/profile/model calls live in killable child processes, never HTTP threads.
"""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import platform
import queue
import tempfile
import threading
import time
import types
import uuid
from typing import Any

from .model_loader import sha256
from .preprocessing import atomic_json
from .video_input import VideoLimits
from .download_face_model import MODEL_PATH
from .performance import counter, observe, summarize, window_timing, rss


class StreamStopped(Exception):
    pass


class NoFaces(Exception):
    pass


def reuse_retina_priors(model):
    """Cache only shape-dependent reference anchors, without editing the package.

    The pinned reference rebuilds identical CPU prior tensors for every frame.
    Give this instance the exact reference function with one private global
    substitution. All transforms, network calls, decoding, NMS and rounding
    remain the reference's own code. Its decode functions do not mutate priors.
    Keep at most two shapes; CPU tensors still follow the original .to(device).
    """
    original = model.predict_jsons.__func__
    reference_priorbox = original.__globals__['priorbox']

    @lru_cache(maxsize=2)
    def cached(min_sizes, steps, clip, image_size):
        return reference_priorbox(min_sizes, steps, clip, image_size)

    def priorbox(min_sizes, steps, clip, image_size):
        return cached(tuple(tuple(v) for v in min_sizes), tuple(steps), clip, tuple(image_size))

    function = types.FunctionType(original.__code__, {**original.__globals__, 'priorbox':priorbox},
                                  original.__name__, original.__defaults__, original.__closure__)
    model.predict_jsons = types.MethodType(function, model)


def save_report(output, report):
    summarize(report)
    rows = report['windows']
    decided = [r for r in rows if r['outcome']=='decided']
    delays = sorted(r['latency_seconds'] for r in decided if 'latency_seconds' in r)
    report['summary'] = dict(window_count=len(rows), decided=len(decided),
        skipped=sum(r['outcome']=='skipped' for r in rows),
        unable_to_decide=sum(r['outcome']=='unable_to_decide' for r in rows),
        cancelled=sum(r['outcome']=='cancelled' for r in rows),
        failed=sum(r['outcome']=='failed' for r in rows),
        p95_latency_seconds=delays[math.ceil(.95*len(delays))-1] if delays else None,
        p50_latency_seconds=delays[math.ceil(.50*len(delays))-1] if delays else None,
        max_latency_seconds=delays[-1] if delays else None,
        latency_sample_count=len(delays),
        percentile_method='nearest_rank; decided windows only; all dispositions disclosed')
    budget=report.get('resource_budget',{}).get('report_metadata_bytes')
    if budget is not None and report.get('phase')=='failed':
        budget += report['resource_budget']['failure_report_reserve_bytes']
    atomic_json(output/'report.json', report, indent=None,max_bytes=budget)
    temporary = output/'windows.tmp'
    columns = ['index','start_seconds','end_seconds','outcome','decision','score','exit_stage',
               'planned_samples','latency_seconds','reason']
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    # Use the same bounded sharing-violation retry as JSON persistence on Windows.
    import os
    for attempt in range(20):
        try:
            os.replace(temporary, output/'windows.csv')
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def _decoder(connection, video):
    import cv2
    cap = cv2.VideoCapture(video)
    try:
        limits = VideoLimits()
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened() or not math.isfinite(fps) or not 0 < fps <= limits.max_fps or \
                not math.isfinite(count) or count < 1 or count != int(count) or count/fps > limits.max_seconds or \
                width <= 0 or height <= 0 or width*height > limits.max_pixels:
            raise ValueError('Invalid or unsupported MP4 metadata')
        connection.send(('ok', dict(fps=fps, frame_count=int(count), width=width, height=height,
                                    duration_seconds=count/fps,decoder_rss=rss())))
        decoded = 0
        while True:
            command, value = connection.recv()
            if command == 'close':
                return
            target = int(count)-1 if command == 'finish' else value
            image = None
            decode_began=time.perf_counter()
            before=decoded
            while decoded <= target:
                ok, image = cap.read()
                if not ok:
                    raise ValueError('MP4 ended before its declared frame count')
                decoded += 1
                if image.shape[:2] != (height, width):
                    raise ValueError('Decoded dimensions differ from metadata')
            if command == 'finish':
                ok, _ = cap.read()
                if ok:
                    raise ValueError('MP4 contains more frames than declared')
                image=None
            else:
                if image is None:
                    raise ValueError('Non-monotonic source frame request')
            connection.send(('ok',dict(image=image,decode_seconds=time.perf_counter()-decode_began,
                decoded_frames=decoded-before,decoder_rss=rss())))
    finally:
        cap.release()


class OfficialWindowEngine:
    """Reuse verified geometry and grouping; only transport/lifetime differ."""
    def __init__(self, output, device, mode, face_model, cache_max_bytes, policy=None,
                 window_seconds=4., sample_hz=2., configs=None, checkpoints=None,
                 stride_seconds=None, window_contract_version=1):
        import torch
        from .adapters import FrameDetectorAdapter, bind_face_profile
        from .face_profiles import build_faces
        from .offline import DEFAULT_CONFIGS, source_state
        from .model_loader import REPO_ROOT
        from .preprocessing import preflight, PROFILE_FOR_MODEL
        from .policy import frozen_policy, execution_contract, validate_runtime_input, validate_runtime_bindings

        self.output = Path(output)
        self.budget = cache_max_bytes
        self.device = device
        self.mode = mode
        torch.set_num_threads(4)
        torch.manual_seed(1024)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.policy, policy_hash = frozen_policy(policy)
        names = [s['detector'] for s in self.policy['stages']]
        self.input_contract = execution_contract(names,unit='window',mode=mode,
            window_seconds=window_seconds,sample_hz=sample_hz,
            stride_seconds=stride_seconds,window_contract_version=window_contract_version)
        validate_runtime_input(self.policy, self.input_contract)
        configs, checkpoints = configs or {}, checkpoints or {}
        if (set(configs)|set(checkpoints))-set(names):
            raise ValueError('Overrides name detectors absent from policy')
        self.definitions = preflight(mode, names, Path(face_model) if face_model is not None else MODEL_PATH)
        self.mapping = {n: PROFILE_FOR_MODEL[n] if mode == 'official' else 'yunet_legacy' for n in names}
        self.detectors = {p: build_faces(p, d['assets'], device) for p, d in self.definitions.items()}
        if 'retinaface' in self.detectors:
            reuse_retina_priors(self.detectors['retinaface'].detector)
        # Each native dlib instance has exclusive ownership during a call. Keep
        # independent instances rather than concurrently sharing mutable state.
        self.face_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix='stream-dlib')
        self.dlib_detectors: queue.Queue = queue.Queue()
        if 'dlib81' in self.definitions:
            self.dlib_detectors.put(self.detectors['dlib81'])
            for _ in range(7):
                self.dlib_detectors.put(build_faces('dlib81', self.definitions['dlib81']['assets'], device))
        weights = dict(xception='xception_best.pth', ucf='ucf_best.pth', sbi='FFc23.tar')
        self.adapters = {}
        for name in names:
            adapter = FrameDetectorAdapter(name, Path(configs.get(name, DEFAULT_CONFIGS[name])),
                Path(checkpoints.get(name, REPO_ROOT/'training/weights'/weights[name])), device, batch_size=8,
                sampling=self.input_contract['sampling'], input_profile=self.mapping[name])
            bind_face_profile(adapter, self.definitions[self.mapping[name]],cadence=self.input_contract)
            self.adapters[name] = adapter
        try:
            validate_runtime_bindings(self.policy,{n:a.provenance for n,a in self.adapters.items()})
            for adapter in self.adapters.values():
                adapter.open()
        except Exception:
            self.close()
            raise
        self.provenance = dict(models={n:a.provenance for n,a in self.adapters.items()},
            input_contract=self.input_contract, policy_sha256=policy_hash,
            profiles=self.definitions, model_profiles=self.mapping, policy=self.policy,
            source=source_state(), environment=dict(python=platform.python_version(), torch=torch.__version__,
                cuda_runtime=torch.version.cuda, device=device,
                gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None))
        self.provenance.update(streaming_implementation_sha256=sha256(Path(__file__)),
            preprocessing_execution=dict(dlib_workers=8, dlib_instances='exclusive per call',
                retina_priors='reference CPU tensors cached by complete shape/anchor settings; max 2 shapes'))
        from collections import OrderedDict
        self.prepared_cache = OrderedDict()
        self.prepared_bytes = 0
        self.prepared_lock = threading.Lock()
        self.scratch = self.output/'scratch'
        self.scratch.mkdir(exist_ok=True)

    def infer(self, window, *, full_scores=False):
        import cv2
        import torch
        from .controller import run_cascade

        profile_results: dict[str, Any] = {p:dict(state='not_reached') for p in self.definitions}
        cache: dict[str, Any] = {}
        trace: list[dict] = []
        reusable=window.get('window_contract_version')==2 and bool(window.get('source_id'))
        used = sum(s['image'].nbytes for s in window['samples']) + (self.budget//4 if reusable else 0)
        if used > self.budget:
            raise ValueError('Active raw/crop cache budget exceeded')
        began = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='window_', dir=self.scratch) as temporary:
            folder = Path(temporary)

            def prepare(name):
                nonlocal used
                profile = self.mapping[name]
                if profile in cache:
                    return cache[profile]
                state: dict[str, Any] = dict(state='running', samples=[], valid_frame_ids=[], face_count=0,
                             planned_samples=len(window['samples']), artifact_retention='hash_only')
                profile_results[profile] = state
                started = time.perf_counter()
                groups = []
                detector = self.detectors[profile]
                def detect(sample):
                    key=(window.get('source_id'),profile,sample.get('frame_id',sample['frame_index']),
                         sample.get('raw_pixels_sha256') or hashlib.sha256(sample['image'].tobytes()).hexdigest())
                    with self.prepared_lock:
                        cached=self.prepared_cache.get(key) if reusable else None
                        if cached is not None:
                            self.prepared_cache.move_to_end(key)
                            return cached[0]
                    instance = self.dlib_detectors.get() if profile == 'dlib81' else detector
                    try:
                        faces = instance(sample['image'])
                        size=sum(crop.nbytes for crop,_ in faces)
                        if reusable and size>self.budget//16:
                            raise ValueError('Per-frame face geometry exceeds transient crop budget')
                        # Own crop storage: a sliced view must not pin an entire
                        # raw source frame outside the accounted crop ceiling.
                        result=([(crop.copy(),metadata) for crop,metadata in faces],
                                getattr(instance, 'last_rejection_reason', 'no_valid_face'))
                        with self.prepared_lock:
                            while self.prepared_cache and (self.prepared_bytes+size>self.budget//4 or len(self.prepared_cache)>=4096):
                                _,(_,old_size)=self.prepared_cache.popitem(last=False)
                                self.prepared_bytes-=old_size
                            if reusable and size<=self.budget//4:
                                # Retain only geometry output, keyed by exact source pixels/profile.
                                self.prepared_cache[key]=(result,size)
                                self.prepared_bytes+=size
                        return result
                    finally:
                        if profile == 'dlib81':
                            self.dlib_detectors.put(instance)

                def bounded_detection():
                    for offset in range(0,len(window['samples']),8):
                        batch=window['samples'][offset:offset+8]
                        yield from (self.face_pool.map(detect,batch) if profile=='dlib81' else map(detect,batch))
                detected = bounded_detection()
                for sample, (faces, rejection_reason) in zip(window['samples'], detected):
                    image = sample['image']
                    paths = []
                    details = []
                    for j, (crop, metadata) in enumerate(faces):
                        ok, encoded = cv2.imencode('.png', crop)
                        if not ok:
                            raise ValueError('Face PNG encoding failed')
                        used += encoded.nbytes
                        if used > self.budget:
                            raise ValueError('Active raw/crop cache budget exceeded')
                        path = folder/f"{profile}_{sample['frame_index']}_{j}.png"
                        path.write_bytes(encoded.tobytes())
                        paths.append(str(path))
                        details.append(dict(**metadata, sha256=sha256(path), shape=list(crop.shape)))
                    entry = dict(frame_index=sample['frame_index'], timestamp_seconds=sample['timestamp_seconds'],
                                 raw_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                 faces=details, face_count=len(faces))
                    if paths:
                        groups.append(dict(frame_index=sample['frame_index'], faces=paths))
                        state['valid_frame_ids'].append(sample['frame_index'])
                    else:
                        entry['reason'] = rejection_reason
                    state['samples'].append(entry)
                state.update(state='used' if groups else 'no_valid_faces', valid_face_frames=len(groups),
                    face_count=sum(len(g['faces']) for g in groups), seconds=time.perf_counter()-started)
                if not groups:
                    raise NoFaces('Reached profile has no valid faces: '+profile)
                cache[profile] = groups
                return groups

            engine = self
            class ReachedAdapter:
                def __init__(self, name):
                    self.name = name
                def infer(self, records):
                    prepared_started = time.perf_counter()
                    groups = prepare(self.name)
                    prepare_seconds = time.perf_counter()-prepared_started
                    enriched = [dict(r, frames=[p for g in groups for p in g['faces']], face_groups=groups)
                                for r in records]
                    scores,timing = engine.adapters[self.name].infer(enriched)
                    timing['prepare_seconds'] = prepare_seconds
                    return scores,timing

            try:
                if full_scores:
                    rows=[]
                    for name in self.adapters:
                        try:
                            scores,timing=ReachedAdapter(name).infer([dict(video_name='window_'+str(window['index']),label=None)])
                            if len(scores)!=1 or scores[0].video_name!='window_'+str(window['index']) or scores[0].label is not None or \
                                    not math.isfinite(scores[0].score) or not 0<=scores[0].score<=1 or scores[0].frame_count<1:
                                raise ValueError('Invalid full-score adapter output')
                            rows.append(dict(detector=name,status='ok',score=scores[0].score,frames=scores[0].frame_count,timing=timing))
                        except Exception as error:
                            rows.append(dict(detector=name,status='no_faces' if isinstance(error,NoFaces) else 'failed',
                                             score=None,frames=0,reason=str(error),timing={}))
                    return dict(score_rows=rows,profiles=profile_results,planned_samples=len(window['samples']))
                result = run_cascade([dict(video_name='window_'+str(window['index']), label=None)],
                    self.policy, {n:ReachedAdapter(n) for n in self.adapters}, on_trace=lambda rows:trace.__setitem__(slice(None),rows))
                decision = result['decisions'][0]
                outcome = dict(outcome='decided', decision=decision['decision'], score=decision['score'],
                    exit_stage=decision['exit_stage'], trace=result['trace'], stages=result['stages'])
            except NoFaces as error:
                outcome = dict(outcome='unable_to_decide', decision=None, score=None, exit_stage=None,
                               reason=str(error), trace=trace, stages=[])
            except Exception as error:
                # Preserve partial profile/trace evidence across the RPC failure.
                error.streaming_evidence = dict(profiles=profile_results, trace=trace)  # type: ignore[attr-defined]
                raise
        outcome.update(profiles=profile_results, planned_samples=len(window['samples']), label=None,
                       processing_seconds=time.perf_counter()-began, active_cache_peak_bytes=used,
                       gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device)
                           if str(self.device).startswith('cuda') else None,
                       gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device)
                           if str(self.device).startswith('cuda') else None,
                       worker_rss=rss())
        outcome['prepared_cache_bytes']=self.prepared_bytes
        outcome['prepared_cache_entries']=len(self.prepared_cache)
        return outcome

    def close(self):
        self.face_pool.shutdown(wait=True)
        for adapter in self.adapters.values():
            adapter.close()
        self.detectors.clear()


def _child(connection, kind, args):
    engine = None
    try:
        if kind == 'decode':
            _decoder(connection, *args)
            return
        engine = OfficialWindowEngine(*args)
        connection.send(('ok', engine.provenance))
        while True:
            command, window = connection.recv()
            if command == 'close':
                return
            connection.send(('ok', engine.infer(window)))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        try:
            connection.send(('error', dict(error=str(error), evidence=getattr(error, 'streaming_evidence', {}))))
        except (OSError, EOFError):
            pass
    finally:
        if engine is not None:
            engine.close()
        connection.close()


class Peer:
    def __init__(self, kind, args, timeout, stop_event=None, *, target=_child):
        self.timeout = timeout
        self.stop_event = stop_event
        ctx = mp.get_context('spawn')
        self.connection, child = ctx.Pipe()
        self.process = ctx.Process(target=target, args=(child, kind, args), daemon=True)
        self.process.start()
        child.close()
        try:
            self.initial = self.receive()
        except BaseException:
            self.close()
            raise

    def receive(self):
        deadline = time.monotonic()+self.timeout
        while not self.connection.poll(.025):
            if self.stop_event is not None and self.stop_event.is_set():
                raise StreamStopped()
            if time.monotonic() >= deadline:
                self.close()
                raise TimeoutError('Native streaming worker timed out and was terminated')
            if not self.process.is_alive():
                raise RuntimeError('Streaming worker exited without a result')
        try:
            status, value = self.connection.recv()
        except EOFError as error:
            raise RuntimeError('Streaming worker disconnected') from error
        if status != 'ok':
            failure = ValueError(value['error'])
            failure.streaming_evidence = value.get('evidence', {})  # type: ignore[attr-defined]
            raise failure
        return value

    def call(self, command, value=None):
        self.connection.send((command, value))
        return self.receive()

    def close(self):
        if self.process.is_alive():
            try:
                self.connection.send(('close', None))
            except (OSError, EOFError):
                pass
            self.process.join(.3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
        self.connection.close()


class VideoSource:
    def __init__(self, video, *, stop_event, **kwargs):
        self.peer = Peer('decode', (str(video),), VideoLimits().decode_timeout_seconds, stop_event)
        self.info = self.peer.initial
        self.measurements=dict(counter(),decoded_frames=0,ipc_image_bytes=0,
                               scope='child local decode/validation; excludes IPC; not additive with parent read')
    def _call(self,command,value=None):
        result=self.peer.call(command,value)
        observe(self.measurements,result['decode_seconds'])
        self.measurements['decoded_frames']+=result['decoded_frames']
        self.measurements['rss']=result['decoder_rss']
        if result['image'] is not None:self.measurements['ipc_image_bytes']+=result['image'].nbytes
        return result['image']
    def read(self, index):
        return self._call('read', index)
    def finish(self):
        return self._call('finish')
    def close(self):
        self.peer.close()


class ModelWorker:
    def __init__(self, output, *, device, preprocessing_mode, face_model, profile_timeout,
                 cache_max_bytes, stop_event, policy=None, window_seconds=4., sample_hz=2.,
                 configs=None, checkpoints=None, stride_seconds=None, window_contract_version=1, **kwargs):
        self.peer = Peer('infer', (str(output), device, preprocessing_mode, str(face_model) if face_model else None,
                         cache_max_bytes,policy,window_seconds,sample_hz,configs,checkpoints,stride_seconds,window_contract_version),
                         profile_timeout, stop_event)
        self.provenance = self.peer.initial
        # A user stop must let an already-running inference finish.
        self.peer.stop_event = None
    def infer(self, window):
        return self.peer.call('infer', window)
    def close(self):
        self.peer.close()


class StreamingSession:
    def __init__(self, video: Path, output: Path, *, device='cuda', preprocessing_mode='official',
                 face_model=None, profile_timeout=180, cache_max_bytes=512*1024*1024,
                 progress=None, source_factory=VideoSource, worker_factory=ModelWorker,
                 window_seconds=4., sample_hz=2., original_filename=None, policy=None,policy_path=None,
                 configs=None,checkpoints=None, analysis_mode="stream",stride_seconds=None,
                 event_source_factory=None, max_windows=10000,max_session_seconds=3600.,max_events=200000,
                 max_report_bytes=64*1024*1024, max_window_metadata_bytes=512*1024, **kwargs):
        if not math.isfinite(profile_timeout) or profile_timeout <= 0 or cache_max_bytes < 1 or \
                not math.isfinite(window_seconds) or window_seconds <= 0 or not math.isfinite(sample_hz) or sample_hz <= 0:
            raise ValueError('Invalid streaming budget or cadence')
        if any(type(v) is not int or v<1 for v in (max_report_bytes,max_window_metadata_bytes)) or max_window_metadata_bytes>512*1024:
            raise ValueError('Invalid metadata budget')
        from .windowing import WindowTimeline
        from .temporal import TemporalAggregator
        if analysis_mode not in ('stream','offline_window'):
            raise ValueError('Unsupported window analysis mode')
        self.event_source_factory=event_source_factory
        self.version=2 if event_source_factory or stride_seconds is not None or analysis_mode=='offline_window' else 1
        self.stride_seconds=window_seconds if stride_seconds is None else stride_seconds
        self.analysis_mode=analysis_mode
        self.timeline_limits=dict(max_windows=max_windows,max_seconds=max_session_seconds,max_events=max_events)
        WindowTimeline('validation','validation',window_seconds=window_seconds,stride_seconds=self.stride_seconds,
                       sample_hz=sample_hz,cache_max_bytes=cache_max_bytes,**self.timeline_limits)
        self.aggregate=TemporalAggregator()
        self.aggregate_ids: set[str] = set()
        self.max_window_metadata_bytes=max_window_metadata_bytes
        self.last_progress_save=0.
        self.video, self.output = video, output
        from .policy import frozen_policy, execution_contract, validate_runtime_input
        selected, policy_hash = frozen_policy(policy,path=policy_path)
        actual = execution_contract([s['detector'] for s in selected['stages']],unit='window',mode=preprocessing_mode,
                                    window_seconds=window_seconds,sample_hz=sample_hz,
                                    stride_seconds=self.stride_seconds,window_contract_version=self.version)
        validate_runtime_input(selected,actual)
        self.settings = dict(device=device, preprocessing_mode=preprocessing_mode, face_model=face_model,
                             profile_timeout=profile_timeout, cache_max_bytes=cache_max_bytes,
                             policy=selected,window_seconds=window_seconds,sample_hz=sample_hz,
                             configs=configs,checkpoints=checkpoints)
        if self.version==2:
            self.settings.update(stride_seconds=self.stride_seconds,window_contract_version=2)
        self.progress = progress or (lambda phase, message: None)
        self.source_factory, self.worker_factory = source_factory, worker_factory
        self.window_seconds, self.sample_hz = window_seconds, sample_hz
        self.stop_event = threading.Event()
        self.condition = threading.Condition(threading.RLock())
        self.rows: list[dict] = []
        self.pending: dict | None = None
        self.collecting: dict | None = None
        self.input_done = False
        self.failure: Exception | None = None
        self.started = 0.
        self.report: dict[str, Any] = dict(schema_version=4, window_contract_version=self.version, analysis_mode=analysis_mode, phase='starting',
            session_id=uuid.uuid4().hex,source_id=None,observed_horizon_seconds=0.,
            policy=selected,policy_sha256=policy_hash,input_contract=actual,
            original_filename=original_filename or video.name, ground_truth=None, deployment_ready=False,
            calibration_status='not_calibrated_for_profile', windows=self.rows,
            window_seconds=window_seconds, sample_hz=sample_hz, stride_seconds=self.stride_seconds,
            sampling=actual['sampling'], preprocessing_mode=preprocessing_mode,
            metrics={k:None for k in ('acc','fnr','fpr','tp','fn','fp','tn')},
            resource_budget=dict(active_cache_bytes=cache_max_bytes, native_call_timeout_seconds=profile_timeout,
                report_metadata_bytes=max_report_bytes,window_metadata_bytes=max_window_metadata_bytes,
                failure_report_reserve_bytes=4*1024*1024,
                max_windows=max_windows,max_session_seconds=max_session_seconds,max_events=max_events,
                retention='complete metadata retained within record/event/time/report budgets; no automatic deletion',
                persisted_artifacts='metadata_only; source upload retained; raw/crop scratch reclaimed per window'),
            warnings=['模擬即時串流；探索性、未校準。Fake score 不是判斷正確的機率。',
                      'Xception/UCF 最大臉；官方 SBI 合格多臉幀內 max，再跨有效幀 mean；無人物追蹤。'])
        self.report['performance'] = dict(schema_version=1, slo_verdict='NOT_EVALUATED',
            percentile_method='nearest_rank; decided windows; grouped by latency_scope; startup excluded',
            source_read=dict(counter(),scope='parent read roundtrip incl IPC/reader scheduling; not pure native decode'),
            admission=dict(counter(),scope='timeline validation/selection; excludes media pacing'),
            persistence=dict(counter(),bytes_written_total=0,
                scope='completed report/CSV writes before this snapshot; excludes current write and fsync guarantee'),
            ipc_raw_payload_bytes_total=0,ipc_scope='sum of inference input image nbytes; not physical copies or RSS',
            latency_by_scope={}, parent_rss=rss(),
            worker_resource_scope='per-result worker RSS samples; GPU process lifetime allocator peaks incl startup; unavailable on fixtures/CPU')
        self.report['performance']['progress_checkpoint_seconds']=.25
        self.report['performance']['metadata_budget_scope']='serialized UTF-8 report; not Python heap or process RSS; failure reserve is additional'
        self.report['performance']['running_windows_peak']=0
        self.report['performance']['pending_windows_peak']=0

    def stop(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()

    def _save(self):
        """Called under the condition lock; artifacts precede observable status."""
        from .temporal import TERMINAL
        for row in self.rows:
            if row['outcome'] in TERMINAL and row['window_id'] not in self.aggregate_ids:
                self.aggregate.update(row)
                self.aggregate_ids.add(row['window_id'])
        phase=self.report['phase']
        perf=self.report['performance']
        perf['running_windows_peak']=max(perf['running_windows_peak'],sum(r['outcome']=='processing' for r in self.rows))
        perf['pending_windows_peak']=max(perf['pending_windows_peak'],int(self.pending is not None))
        perf['aggregate_records']=len(self.aggregate.results)
        reason={'complete':'eof','stopped':'user_stop','failed':'failure'}.get(phase)
        previous_reason=self.aggregate.final_reason
        self.report['aggregate']=self.aggregate.snapshot(self.report['observed_horizon_seconds'],
            pending=sum(r['outcome'] not in TERMINAL for r in self.rows),finalization_reason=reason)
        began=time.perf_counter()
        try:
            save_report(self.output, self.report)
        except Exception:
            # A failed write must not commit an aggregator finalization that
            # prevents a subsequent failure report with the same evidence.
            self.aggregate.final_reason=previous_reason
            raise
        observe(self.report['performance']['persistence'],time.perf_counter()-began)
        self.report['performance']['persistence']['bytes_written_total'] += sum(
            (self.output/name).stat().st_size for name in ('report.json','windows.csv'))
        self.last_progress_save=time.monotonic()
        self.report['performance']['last_report_bytes']=(self.output/'report.json').stat().st_size

    def _phase(self, phase, message):
        with self.condition:
            self.report.update(phase=phase, message=message)
            if phase in ('complete','stopped','failed'):
                self.report['performance']['aggregate_evidence_serialized_bytes']=len(
                    json.dumps(self.aggregate.results,ensure_ascii=False).encode('utf-8'))
            try:
                self._save()
            except Exception as error:
                if phase not in ('complete','stopped'):
                    raise
                self.failure=error
                phase='failed'
                message=str(error)
                self.report.update(phase=phase,message=message)
                self._save()
            atomic_json(self.output/'status.json', dict(phase=phase, message=message))
        self.progress(phase, message)

    def _wait_media(self, seconds):
        while not self.stop_event.is_set():
            remaining = self.started+seconds-time.monotonic()
            if remaining <= 0:
                break
            self.stop_event.wait(remaining)
        if self.stop_event.is_set():
            raise StreamStopped()
        if self.failure is not None:
            raise self.failure

    def _consume(self, worker):
        try:
            self._consume_loop(worker)
        except Exception as error:
            # Persistence failures must also stop the producer; a dead consumer
            # must never leave a session marked complete with pending rows.
            with self.condition:
                self.failure = error
                for row in self.rows:
                    if row['outcome'] == 'processing':
                        row.update(outcome='failed', reason=str(error), decision=None)
                self.condition.notify_all()

    def _consume_loop(self, worker):
        while True:
            with self.condition:
                while self.pending is None and not self.input_done and not self.stop_event.is_set():
                    self.condition.wait(.05)
                if self.pending is None:
                    return
                window = self.pending
                self.pending = None
                row = self.rows[window['index']]
                if self.stop_event.is_set():
                    row.update(outcome='cancelled', reason='user_stop')
                    self._save()
                    continue
                row['outcome'] = 'processing'
                queue_seconds=max(0.,time.monotonic()-self.started-window.get('ready_elapsed_seconds',time.monotonic()-self.started))
                self._save()
            try:
                infer_began=time.perf_counter()
                result = worker.infer(window)
                roundtrip=time.perf_counter()-infer_began
                with self.condition:
                    if len(json.dumps(result,ensure_ascii=False,allow_nan=False).encode('utf-8'))>self.max_window_metadata_bytes:
                        raise ValueError('Window metadata budget exceeded before evidence acceptance')
                    self.report['performance']['ipc_raw_payload_bytes_total'] += sum(s['image'].nbytes for s in window['samples'])
                    if result.get('outcome') not in ('decided','unable_to_decide'):
                        raise ValueError('Window executor returned an invalid disposition')
                    for key in ('window_id','source_id','session_id','index','window_contract_version',
                                'start_seconds','end_seconds','observed_end_seconds','policy_sha256','input_contract'):
                        if key in result and result[key]!=row.get(key):
                            raise ValueError('Window executor changed immutable '+key)
                    # Validate terminal evidence before merging. Once accepted,
                    # persistence errors fail the session, not the classifier's
                    # already finalized decision.
                    self.aggregate.update(dict(row,**result))
                    row.update(result)
                    row['timing']=window_timing(result,queue_seconds,roundtrip)
                    # Persistence is part of latency. First save makes result readable;
                    # second persists the measured availability timestamp itself.
                    persist_began=time.perf_counter()
                    self._save()
                    row['timing']['persistence_seconds']=time.perf_counter()-persist_began
                    elapsed = time.monotonic()-self.started
                    origin=(window.get('ready_elapsed_seconds',elapsed) if self.analysis_mode=='offline_window'
                            else row['end_seconds'])
                    row.update(result_available_elapsed_seconds=elapsed,
                               latency_seconds=max(0., elapsed-origin),
                               latency_scope='since_window_admission' if self.analysis_mode=='offline_window' else 'since_media_window_end')
                    self._save()
            except Exception as error:
                with self.condition:
                    if row['outcome']=='processing':
                        row.update(outcome='failed', reason=str(error), decision=None,
                                   evidence=getattr(error, 'streaming_evidence', {}))
                    self.failure = error
                    self._save()
                    self.condition.notify_all()
                return

    def _run_events(self):
        from .windowing import WindowTimeline, EventReader, mp4_events, window_record
        self.output.mkdir(parents=True,exist_ok=False)
        source=reader=worker=consumer=timeline=None
        began=time.monotonic()
        def submit(window, cancelled=False):
            with self.condition:
                window.update(policy_sha256=self.report['policy_sha256'],input_contract=self.report['input_contract'])
                window['ready_elapsed_seconds']=time.monotonic()-self.started
                row=window_record(window)
                self.rows.append(row)
                if cancelled:
                    row.update(outcome='cancelled',reason='user_stop' if self.stop_event.is_set() else 'session_failed')
                elif not window['samples']:
                    row.update(outcome='unable_to_decide',reason='no_unique_source_samples')
                else:
                    if self.analysis_mode=='offline_window':
                        # Backpressure: a fast source never replaces offline work.
                        while self.pending is not None or any(r['outcome']=='processing' for r in self.rows):
                            self.condition.wait(.01)
                            if self.failure is not None:
                                raise self.failure
                            if self.stop_event.is_set():
                                row.update(outcome='cancelled',reason='user_stop')
                                raise StreamStopped()
                    elif self.pending is not None:
                        self.rows[self.pending['index']].update(outcome='skipped',reason='overload_replaced_by_newer_window')
                    self.pending=window
                    row['outcome']='waiting'
                self._save()
                self.condition.notify_all()
        try:
            self._phase('starting','驗證來源、凍結 Window Policy 與載入模型')
            if self.event_source_factory is None:
                if self.video.suffix.lower()!='.mp4' or not self.video.is_file() or not 0<self.video.stat().st_size<=VideoLimits().max_bytes:
                    raise ValueError('Invalid MP4 input or size')
                self.report['input_sha256']=sha256(self.video)
                source=self.source_factory(self.video,stop_event=self.stop_event)
                source_id=self.report['input_sha256']
            else:
                source=self.event_source_factory(self.video,stop_event=self.stop_event)
                source_id=source.info.get('source_id',self.report['session_id'])
                reader=EventReader(source,self.stop_event,self.settings['profile_timeout'])
            if not isinstance(source_id,str) or not 0<len(source_id)<=256:
                raise ValueError('Invalid source identity')
            self.report.update(source=dict(source.info),source_id=source_id)
            # Disjoint ceilings cover timeline, producer batch, waiting/running
            # raw samples, plus worker raw/crops. Image references are shared
            # within the producer; the native worker's transport is a copy.
            budget=self.settings['cache_max_bytes']//4
            if budget<1:
                raise ValueError('Active cache budget too small')
            timeline=WindowTimeline(source_id,self.report['session_id'],window_seconds=self.window_seconds,
                stride_seconds=self.stride_seconds,sample_hz=self.sample_hz,cache_max_bytes=budget,**self.timeline_limits)
            self.report['resource_budget'].update(timeline_raw_bytes=budget,inference_cache_bytes=budget,
                max_windows=timeline.max_windows,max_session_seconds=timeline.max_seconds,max_events=timeline.max_events,
                source_diagnostics=1000,max_active_windows=128,reader_threads=1,
                reader_timeout_policy='quarantine noncooperative source until it returns')
            worker=self.worker_factory(self.output,stop_event=self.stop_event,**dict(self.settings,cache_max_bytes=budget))
            self.report.update(provenance=worker.provenance,startup_seconds=time.monotonic()-began)
            atomic_json(self.output/'manifest.json',{k:v for k,v in self.report.items() if k!='windows'})
            self.started=time.monotonic()
            self.report['playback_started_unix_seconds']=time.time()
            self._phase('playing','Offline Window 分析' if self.analysis_mode=='offline_window' else '原速 Window 串流')
            consumer=threading.Thread(target=self._consume,args=(worker,),daemon=True)
            consumer.start()
            events=mp4_events(source,self.sample_hz) if reader is None else None
            while True:
                if self.stop_event.is_set():
                    raise StreamStopped()
                if self.failure is not None:
                    raise self.failure
                read_began=time.perf_counter()
                event=reader.read() if reader else next(events)
                read_elapsed=time.perf_counter()-read_began
                with self.condition:
                    if hasattr(source,'measurements'):
                        self.report['performance']['native_decode']=dict(source.measurements)
                    observe(self.report['performance']['source_read'],read_elapsed)
                seconds=event.get('sampling_tick_seconds',event.get('timestamp_seconds',event.get('seconds',timeline.watermark)))
                if (isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or
                        seconds<0 or seconds>timeline.max_seconds):
                    raise ValueError('Source duration budget exceeded or invalid event time')
                if self.analysis_mode=='stream':
                    self._wait_media(seconds)
                admission_began=time.perf_counter()
                windows=timeline.accept(event)
                admission_elapsed=time.perf_counter()-admission_began
                with self.condition:
                    observe(self.report['performance']['admission'],admission_elapsed)
                    self.report['observed_horizon_seconds']=timeline.watermark
                    self.report['media_seconds']=timeline.watermark
                    self.report['source_events']=timeline.events
                    self.report['source_diagnostics']=list(timeline.diagnostics)
                    self.report['performance'].update(timeline_raw_peak_bytes=timeline.peak_raw_bytes,
                        timeline_identity_records=len(timeline.identities),
                        window_sample_metadata=timeline.published_samples,
                        emitted_window_batch_peak=max(self.report['performance'].get('emitted_window_batch_peak',0),len(windows)))
                for window in windows:
                    submit(window)
                with self.condition:
                    # Terminal transitions are always persisted synchronously.
                    # Coalesce progress at the next eligible event. A crash may
                    # lose uncheckpointed progress, never fabricate observation;
                    # slow/idle sources do not promise a wall-clock flush deadline.
                    if time.monotonic()-self.last_progress_save>=.25:
                        self._save()
                        self.last_progress_save=time.monotonic()
                if event['type']=='end':
                    break
            self._phase('draining','來源已結束，等待保留的窗完成')
        except StreamStopped:
            self._phase('stopping','停止接收影格，完成正在推論的窗')
        except Exception as error:
            self.failure=error
        finally:
            if timeline is not None:
                self.report['source_diagnostics']=list(timeline.diagnostics)
                # Only committed observation time contributes to stop/failure tails.
                if not timeline.closed and timeline.watermark>0:
                    try:
                        for window in timeline.accept(dict(type='end',seconds=timeline.watermark)):
                            submit(window,cancelled=True)
                    except Exception as error:
                        if self.failure is None:self.failure=error
            with self.condition:
                self.input_done=True
                if self.stop_event.is_set() or self.failure:
                    for row in self.rows:
                        if row['outcome'] in ('collecting','waiting'):
                            row.update(outcome='cancelled',reason='user_stop' if self.stop_event.is_set() else 'session_failed')
                    self.pending=None
                self.condition.notify_all()
            if consumer is not None:consumer.join()
            if worker is not None:worker.close()
            if reader is not None:reader.close()
            elif source is not None:source.close()
        phase='failed' if self.failure else 'stopped' if self.stop_event.is_set() else 'complete'
        self.report['total_seconds']=time.monotonic()-began
        if self.failure:atomic_json(self.output/'error.json',dict(error=str(self.failure)))
        self._phase(phase,str(self.failure) if self.failure else 'Window 分析完成')
        return self.report

    def run(self):
        if self.version==2:
            return self._run_events()
        self.output.mkdir(parents=True, exist_ok=False)
        source = worker = None
        consumer = None
        began = time.monotonic()
        try:
            self._phase('starting', '驗證影片、載入模型與暖機；尚未開始播放')
            if self.video.suffix.lower() != '.mp4' or not self.video.is_file() or not 0 < self.video.stat().st_size <= VideoLimits().max_bytes:
                raise ValueError('Invalid MP4 input or size')
            self.report['input_sha256'] = sha256(self.video)
            self.report['source_id']=self.report['input_sha256']
            source = self.source_factory(self.video, stop_event=self.stop_event)
            raw_reserve = 2*math.ceil(self.window_seconds*self.sample_hz)*source.info.get('width',0)*source.info.get('height',0)*3
            worker_settings = dict(self.settings, cache_max_bytes=self.settings['cache_max_bytes']-raw_reserve)
            if worker_settings['cache_max_bytes'] <= 0:
                raise ValueError('Active cache budget cannot hold collecting/waiting raw frames')
            self.report['resource_budget']['inference_cache_bytes'] = worker_settings['cache_max_bytes']
            worker = self.worker_factory(self.output, stop_event=self.stop_event, **worker_settings)
            self.report.update(source=source.info, provenance=worker.provenance,
                               startup_seconds=time.monotonic()-began)
            atomic_json(self.output/'manifest.json', {k:v for k,v in self.report.items() if k!='windows'})
            if self.stop_event.is_set():
                raise StreamStopped()
            self.started = time.monotonic()
            self.report['playback_started_unix_seconds'] = time.time()
            self._phase('playing', '原速串流中：4 秒一窗，2 幀／秒')
            consumer = threading.Thread(target=self._consume, args=(worker,), daemon=True)
            consumer.start()
            duration = source.info['duration_seconds']
            count = int(math.ceil(duration/self.window_seconds-1e-10))
            if (duration>self.timeline_limits['max_seconds'] or count>self.timeline_limits['max_windows'] or
                    math.ceil(duration*self.sample_hz)>self.timeline_limits['max_events']):
                raise ValueError('Legacy session duration/window/sample record budget exceeded')
            previous_index = -1
            for i in range(count):
                start, end = i*self.window_seconds, min((i+1)*self.window_seconds, duration)
                window: dict[str, Any] = dict(index=i, start_seconds=start, end_seconds=end, samples=[])
                with self.condition:
                    self.rows.append(dict(index=i, start_seconds=start, end_seconds=end,
                        nominal_end_seconds=(i+1)*self.window_seconds,observed_end_seconds=start,
                        window_contract_version=1,source_id=self.report['source_id'],session_id=self.report['session_id'],
                        window_id=f"{self.report['session_id']}:{i}",source_frames=[],tail=end<(i+1)*self.window_seconds,
                        outcome='collecting', decision=None, planned_samples=0))
                    window.update({k:v for k,v in self.rows[i].items() if k not in ('source_frames','outcome','decision','planned_samples')})
                    window.update(policy_sha256=self.report['policy_sha256'],input_contract=self.report['input_contract'])
                    self.rows[i].update(policy_sha256=self.report['policy_sha256'],input_contract=self.report['input_contract'])
                    self.collecting = window
                for j in range(int(math.ceil((end-start)*self.sample_hz-1e-10))):
                    timestamp = start+j/self.sample_hz
                    self._wait_media(timestamp)
                    index = min(int(timestamp*source.info['fps']), source.info['frame_count']-1)
                    if index == previous_index:
                        continue
                    read_began=time.perf_counter()
                    image = source.read(index)
                    read_elapsed=time.perf_counter()-read_began
                    with self.condition:
                        observe(self.report['performance']['source_read'],read_elapsed)
                        if hasattr(source,'measurements'):
                            self.report['performance']['native_decode']=dict(source.measurements)
                    previous_index = index
                    window['samples'].append(dict(frame_index=index, timestamp_seconds=index/source.info['fps'], image=image))
                    with self.condition:
                        self.rows[i]['source_frames'].append(dict(frame_id=str(index),frame_index=index,
                            timestamp_seconds=index/source.info['fps'],sampling_tick_seconds=timestamp,
                            timestamp_basis='frame_index_over_fps',raw_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
                        self.report['observed_horizon_seconds']=timestamp
                        self.rows[i]['observed_end_seconds']=timestamp
                        window['samples'][-1].update(self.rows[i]['source_frames'][-1])
                        window['observed_end_seconds']=timestamp
                    with self.condition:
                        self.rows[i]['planned_samples'] = len(window['samples'])
                    if sum(s['image'].nbytes for s in window['samples']) > self.settings['cache_max_bytes']:
                        raise ValueError('Collecting window exceeds active raw cache budget')
                self._wait_media(end)
                with self.condition:
                    self.report['observed_horizon_seconds']=end
                    self.rows[i]['observed_end_seconds']=end
                    window['observed_end_seconds']=end
                    row = self.rows[i]
                    row['planned_samples'] = len(window['samples'])
                    self.collecting = None
                    if self.stop_event.is_set():
                        row.update(outcome='cancelled', reason='user_stop')
                        raise StreamStopped()
                    if not window['samples']:
                        row.update(outcome='unable_to_decide', reason='no_unique_source_samples')
                        self._save()
                        continue
                    if self.pending is not None:
                        old = self.rows[self.pending['index']]
                        old.update(outcome='skipped', reason='overload_replaced_by_newer_window')
                    self.pending = window
                    window['ready_elapsed_seconds']=time.monotonic()-self.started
                    row['outcome'] = 'waiting'
                    self.report['media_seconds'] = end
                    self._save()
                    self.condition.notify_all()
            source.finish()
            if hasattr(source,'measurements'):
                self.report['performance']['native_decode']=dict(source.measurements)
            self._phase('draining', '影片已結束，等待保留的時間窗完成')
        except StreamStopped:
            self._phase('stopping', '停止接收影格，完成正在推論的窗')
        except Exception as error:
            self.failure = error
        finally:
            with self.condition:
                self.input_done = True
                for row in self.rows:
                    if row['outcome'] in ('collecting', 'waiting') and (self.stop_event.is_set() or self.failure):
                        row.update(outcome='cancelled', reason='user_stop' if self.stop_event.is_set() else 'session_failed')
                if self.stop_event.is_set() or self.failure:
                    self.pending = None
                self.condition.notify_all()
            if consumer is not None:
                consumer.join()  # each native inference has an enforced worker timeout
            if worker is not None:
                worker.close()
            if source is not None:
                source.close()
        phase = 'failed' if self.failure else 'stopped' if self.stop_event.is_set() else 'complete'
        self.report['total_seconds'] = time.monotonic()-began
        if self.failure:
            atomic_json(self.output/'error.json', dict(error=str(self.failure)))
        self._phase(phase, str(self.failure) if self.failure else '串流已停止' if phase=='stopped' else '串流完成')
        return self.report
