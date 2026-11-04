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
from typing import Any

from .model_loader import sha256
from .preprocessing import atomic_json
from .video_input import VideoLimits
from .download_face_model import MODEL_PATH


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
        percentile_method='nearest_rank; decided windows only; all dispositions disclosed')
    atomic_json(output/'report.json', report, indent=None)
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
                                    duration_seconds=count/fps)))
        decoded = 0
        while True:
            command, value = connection.recv()
            if command == 'close':
                return
            target = int(count)-1 if command == 'finish' else value
            image = None
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
                connection.send(('ok', None))
            else:
                if image is None:
                    raise ValueError('Non-monotonic source frame request')
                connection.send(('ok', image))
    finally:
        cap.release()


class OfficialWindowEngine:
    """Reuse verified geometry and grouping; only transport/lifetime differ."""
    def __init__(self, output, device, mode, face_model, cache_max_bytes):
        import torch
        from .adapters import FrameDetectorAdapter
        from .face_profiles import build_faces
        from .offline import DEFAULT_CONFIGS, source_state
        from .model_loader import REPO_ROOT
        from .preprocessing import preflight, PROFILE_FOR_MODEL
        from .video import POLICIES
        from .policy import validate_policy

        self.output = Path(output)
        self.budget = cache_max_bytes
        self.device = device
        self.mode = mode
        torch.set_num_threads(4)
        torch.manual_seed(1024)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        names = ['xception', 'ucf', 'sbi']
        self.policy = validate_policy(json.loads(POLICIES['early'].read_text(encoding='utf-8')),
                                      'uploaded_video', allow_exploratory=True)
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
            adapter = FrameDetectorAdapter(name, DEFAULT_CONFIGS[name],
                REPO_ROOT/'training/weights'/weights[name], device, batch_size=8,
                sampling='paced_2Hz_nonoverlapping_4s', input_profile=self.mapping[name])
            adapter.provenance['face_profile'] = self.definitions[self.mapping[name]]
            adapter.provenance['preprocessing']['aggregation'] = 'max_face_then_mean_frame'
            adapter.provenance['preprocessing_sha256'] = hashlib.sha256(json.dumps(dict(
                tensor=adapter.provenance['preprocessing'], face_profile=adapter.provenance['face_profile']),
                sort_keys=True).encode()).hexdigest()
            expected = self.policy.get('preprocessing_sha256', {}).get(name)
            if expected is not None and expected != adapter.provenance['preprocessing_sha256']:
                raise ValueError('Policy preprocessing binding mismatch: '+name)
            adapter.open()
            self.adapters[name] = adapter
        self.provenance = dict(models={n:a.provenance for n,a in self.adapters.items()},
            profiles=self.definitions, model_profiles=self.mapping, policy=self.policy,
            source=source_state(), environment=dict(python=platform.python_version(), torch=torch.__version__,
                cuda_runtime=torch.version.cuda, device=device,
                gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None))
        self.provenance.update(streaming_implementation_sha256=sha256(Path(__file__)),
            preprocessing_execution=dict(dlib_workers=8, dlib_instances='exclusive per call',
                retina_priors='reference CPU tensors cached by complete shape/anchor settings; max 2 shapes'))
        self.scratch = self.output/'scratch'
        self.scratch.mkdir(exist_ok=True)

    def infer(self, window):
        import cv2
        import torch
        from .controller import run_cascade

        profile_results: dict[str, Any] = {p:dict(state='not_reached') for p in self.definitions}
        cache: dict[str, Any] = {}
        trace: list[dict] = []
        used = sum(s['image'].nbytes for s in window['samples'])
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
                    instance = self.dlib_detectors.get() if profile == 'dlib81' else detector
                    try:
                        faces = instance(sample['image'])
                        return faces, getattr(instance, 'last_rejection_reason', 'no_valid_face')
                    finally:
                        if profile == 'dlib81':
                            self.dlib_detectors.put(instance)

                detected = (self.face_pool.map(detect, window['samples']) if profile == 'dlib81'
                            else map(detect, window['samples']))
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
                    groups = prepare(self.name)
                    enriched = [dict(r, frames=[p for g in groups for p in g['faces']], face_groups=groups)
                                for r in records]
                    return engine.adapters[self.name].infer(enriched)

            try:
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
                           if str(self.device).startswith('cuda') else None)
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
    def read(self, index):
        return self.peer.call('read', index)
    def finish(self):
        return self.peer.call('finish')
    def close(self):
        self.peer.close()


class ModelWorker:
    def __init__(self, output, *, device, preprocessing_mode, face_model, profile_timeout,
                 cache_max_bytes, stop_event, **kwargs):
        self.peer = Peer('infer', (str(output), device, preprocessing_mode, str(face_model), cache_max_bytes),
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
                 window_seconds=4., sample_hz=2., original_filename=None, **kwargs):
        if not math.isfinite(profile_timeout) or profile_timeout <= 0 or cache_max_bytes < 1 or \
                not math.isfinite(window_seconds) or window_seconds <= 0 or not math.isfinite(sample_hz) or sample_hz <= 0:
            raise ValueError('Invalid streaming budget or cadence')
        self.video, self.output = video, output
        self.settings = dict(device=device, preprocessing_mode=preprocessing_mode, face_model=face_model,
                             profile_timeout=profile_timeout, cache_max_bytes=cache_max_bytes)
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
        self.report: dict[str, Any] = dict(schema_version=3, analysis_mode='stream', phase='starting',
            original_filename=original_filename or video.name, ground_truth=None, deployment_ready=False,
            calibration_status='not_calibrated_for_profile', windows=self.rows,
            window_seconds=window_seconds, sample_hz=sample_hz, stride_seconds=window_seconds,
            sampling='paced_media_time_no_future_frames', preprocessing_mode=preprocessing_mode,
            metrics={k:None for k in ('acc','fnr','fpr','tp','fn','fp','tn')},
            resource_budget=dict(active_cache_bytes=cache_max_bytes, native_call_timeout_seconds=profile_timeout,
                persisted_artifacts='metadata_only; source upload retained; raw/crop scratch reclaimed per window'),
            warnings=['模擬即時串流；探索性、未校準。Fake score 不是判斷正確的機率。',
                      'Xception/UCF 最大臉；官方 SBI 合格多臉幀內 max，再跨有效幀 mean；無人物追蹤。'])

    def stop(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()

    def _save(self):
        """Called under the condition lock; artifacts precede observable status."""
        save_report(self.output, self.report)

    def _phase(self, phase, message):
        with self.condition:
            self.report.update(phase=phase, message=message)
            self._save()
            atomic_json(self.output/'status.json', dict(phase=phase, message=message))
        self.progress(phase, message)

    def _wait_media(self, seconds):
        remaining = self.started+seconds-time.monotonic()
        if remaining > 0:
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
                self._save()
            try:
                result = worker.infer(window)
                with self.condition:
                    row.update(result)
                    # Persistence is part of latency. First save makes result readable;
                    # second persists the measured availability timestamp itself.
                    self._save()
                    elapsed = time.monotonic()-self.started
                    row.update(result_available_elapsed_seconds=elapsed,
                               latency_seconds=max(0., elapsed-row['end_seconds']))
                    self._save()
            except Exception as error:
                with self.condition:
                    row.update(outcome='failed', reason=str(error), decision=None,
                               evidence=getattr(error, 'streaming_evidence', {}))
                    self.failure = error
                    self._save()
                    self.condition.notify_all()
                return

    def run(self):
        self.output.mkdir(parents=True, exist_ok=False)
        source = worker = None
        consumer = None
        began = time.monotonic()
        try:
            self._phase('starting', '驗證影片、載入模型與暖機；尚未開始播放')
            if self.video.suffix.lower() != '.mp4' or not self.video.is_file() or not 0 < self.video.stat().st_size <= VideoLimits().max_bytes:
                raise ValueError('Invalid MP4 input or size')
            self.report['input_sha256'] = sha256(self.video)
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
            previous_index = -1
            for i in range(count):
                start, end = i*self.window_seconds, min((i+1)*self.window_seconds, duration)
                window: dict[str, Any] = dict(index=i, start_seconds=start, end_seconds=end, samples=[])
                with self.condition:
                    self.rows.append(dict(index=i, start_seconds=start, end_seconds=end,
                                          outcome='collecting', decision=None, planned_samples=0))
                    self.collecting = window
                for j in range(int(math.ceil((end-start)*self.sample_hz-1e-10))):
                    timestamp = start+j/self.sample_hz
                    self._wait_media(timestamp)
                    index = min(int(timestamp*source.info['fps']), source.info['frame_count']-1)
                    if index == previous_index:
                        continue
                    image = source.read(index)
                    previous_index = index
                    window['samples'].append(dict(frame_index=index, timestamp_seconds=index/source.info['fps'], image=image))
                    with self.condition:
                        self.rows[i]['planned_samples'] = len(window['samples'])
                    if sum(s['image'].nbytes for s in window['samples']) > self.settings['cache_max_bytes']:
                        raise ValueError('Collecting window exceeds active raw cache budget')
                self._wait_media(end)
                with self.condition:
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
                    row['outcome'] = 'waiting'
                    self.report['media_seconds'] = end
                    self._save()
                    self.condition.notify_all()
            source.finish()
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
