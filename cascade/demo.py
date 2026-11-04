"""Local-only MP4 upload demo using the same pipeline as cascade.video."""
import argparse
import json
import mimetypes
import os
from pathlib import Path
import re
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid
import webbrowser
from typing import cast

from .download_face_model import MODEL_PATH
from .offline import write_json
from .video import POLICIES, run_video
from .video_input import VideoLimits
from .streaming import StreamingSession, save_report
from .policy import frozen_policy, execution_contract, validate_runtime_input

PAGE = Path(__file__).resolve().parent/'demo.html'


def read_json(path):
    for attempt in range(20):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.01)


class DemoState:
    def __init__(self, root: Path, device='cuda', face_model: Path = MODEL_PATH, runner=run_video,
                 preprocessing_mode='official', profile_timeout=180, cache_max_bytes=512*1024*1024,
                 stream_factory=StreamingSession, stream_policies=None):
        self.stream_factory = stream_factory
        self.stream_policies = {'early':frozen_policy(path=POLICIES['early'])[0]}
        for name,value in (stream_policies or {}).items():
            if not re.fullmatch('[A-Za-z0-9_-]+',name):raise ValueError('Invalid registered policy name')
            self.stream_policies[name] = frozen_policy(value if isinstance(value,dict) else None,
                path=None if isinstance(value,dict) else Path(value))[0]
        self.sessions: dict[str, StreamingSession] = {}
        self.server_started = time.time()
        from .model_loader import sha256
        self.server_build = sha256(Path(__file__))
        self.workers: list[threading.Thread] = []
        self.preprocessing_mode=preprocessing_mode
        self.profile_timeout=profile_timeout
        self.cache_max_bytes=cache_max_bytes
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.device, self.face_model, self.runner = device, face_model, runner
        self.lock = threading.Lock()
        self.active: str | None = None
        self.jobs: dict[str, dict] = {}
        for path in sorted(self.root.glob('*/job.json'), key=lambda p:p.stat().st_mtime):
            if not re.fullmatch('[0-9a-f]{32}', path.parent.name):
                continue
            try:
                job = json.loads(path.read_text(encoding='utf-8'))
                job.update(id=path.parent.name, phase='failed', message='先前工作已中斷，請重新上傳')
                result = path.parent/'result'
                if job.get('analysis_mode') == 'stream' and (result/'report.json').is_file():
                    report = json.loads((result/'report.json').read_text(encoding='utf-8'))
                    phase = report.get('phase')
                    if phase not in ('complete', 'stopped', 'failed'):
                        phase = 'failed'
                        report.update(phase='failed', message='服務重啟：先前串流已中斷，不自動續跑')
                        for row in report.get('windows', []):
                            if row['outcome'] in ('collecting', 'waiting', 'processing'):
                                row.update(outcome='cancelled', reason='service_interrupted', decision=None)
                        from .preprocessing import atomic_json
                        save_report(result, report)
                        atomic_json(result/'status.json', dict(phase='failed', message=report['message']))
                    job.update(phase=phase, message=report.get('message','串流紀錄'), report=report)
                elif (result/'report.json').is_file() and \
                        json.loads((result/'status.json').read_text(encoding='utf-8'))['phase'] == 'complete':
                    job.update(phase='complete', message='推論完成',
                               report=json.loads((result/'report.json').read_text(encoding='utf-8')))
                elif (result/'error.json').is_file():
                    job['message'] = json.loads((result/'error.json').read_text(encoding='utf-8'))['error']
                self.jobs[job['id']] = job
            except (OSError, ValueError, KeyError):
                continue

    def reserve(self, filename, policy, preprocessing_mode=None, analysis_mode='offline'):
        with self.lock:
            if self.active is not None:
                raise RuntimeError('已有影片正在處理，請等待完成後再上傳')
            selected = None
            if analysis_mode=='stream':
                selected,policy_hash=frozen_policy(self.stream_policies[policy])
                validate_runtime_input(selected,execution_contract([s['detector'] for s in selected['stages']],
                    unit='window',mode=preprocessing_mode or self.preprocessing_mode))
            job_id = uuid.uuid4().hex
            folder = self.root/job_id
            folder.mkdir(exist_ok=False)
            job = dict(id=job_id, filename=filename, policy=policy, preprocessing_mode=preprocessing_mode or self.preprocessing_mode, phase='uploading', message='接收影片')
            job['analysis_mode'] = analysis_mode
            if selected is not None:
                job.update(stream_policy=selected,policy_sha256=policy_hash,policy_id=selected['policy_id'])
            write_json(folder/'job.json', job)
            self.jobs[job_id] = job
            self.active = job_id
            return job_id

    def update(self, job_id, phase, message):
        with self.lock:
            self.jobs[job_id].update(phase=phase, message=message)

    def fail(self, job_id, message):
        with self.lock:
            self.jobs[job_id].update(phase='failed', message=message)
            if self.active == job_id:
                self.active = None

    def snapshot(self, job_id):
        with self.lock:
            job=dict(self.jobs[job_id])
            if job.get('analysis_mode') == 'stream' and (self.root/job_id/'result/report.json').is_file():
                report = read_json(self.root/job_id/'result/report.json')
                # Polling needs counts and traces; full per-frame geometry stays
                # in the downloadable report, avoiding quadratic UI traffic.
                report['windows'] = [dict(row, profiles={name:{k:v for k,v in profile.items() if k!='samples'}
                    for name,profile in row.get('profiles',{}).items()}) for row in report['windows']]
                job['report'] = report
            job['diagnostic_files']=[name for name in ('error.json','manifest.json','preprocessing.json','partial_trace.json')
                if (self.root/job_id/'result'/name).is_file()]
            return job

    def start(self, job_id):
        def work():
            job = self.snapshot(job_id)
            try:
                if job.get('analysis_mode') == 'stream':
                    session = self.stream_factory(self.root/job_id/'input.mp4', self.root/job_id/'result',
                        policy=job['stream_policy'],
                        device=self.device, face_model=self.face_model,
                        preprocessing_mode=job['preprocessing_mode'], profile_timeout=self.profile_timeout,
                        cache_max_bytes=self.cache_max_bytes, original_filename=job['filename'],
                        progress=lambda p,m:self.update(job_id,'finalizing' if p in ('complete','stopped','failed') else p,m))
                    with self.lock:
                        self.sessions[job_id] = session
                    result = session.run()
                    with self.lock:
                        self.jobs[job_id].update(report=result, phase=result['phase'], message=result['message'])
                        self.active = None
                    return
                result = self.runner(self.root/job_id/'input.mp4', self.root/job_id/'result',
                                     policy_path=POLICIES[job['policy']], device=self.device,
                                     face_model=self.face_model, allow_exploratory=True,
                                     original_filename=job['filename'],
                                     preprocessing_mode=job.get('preprocessing_mode',self.preprocessing_mode),
                                     profile_timeout=self.profile_timeout,cache_max_bytes=self.cache_max_bytes,
                                     progress=lambda p, m: self.update(job_id,
                                         'finalizing' if p == 'complete' else p, m))
                with self.lock:
                    self.jobs[job_id].update(report=result, phase='complete', message='推論完成')
            except Exception as error:
                self.fail(job_id, str(error))
            finally:
                with self.lock:
                    if self.active == job_id:
                        self.active = None
                    self.sessions.pop(job_id, None)
        thread = threading.Thread(target=work, daemon=True)
        with self.lock:
            self.workers = [t for t in self.workers if t.is_alive()]
            self.workers.append(thread)
        thread.start()

    def stop(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                return 404
            session = self.sessions.get(job_id)
            if session is None:
                return 409
            session.stop()
            return 202


class DemoServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can permit an old and a new server to bind the same
    # port, creating a new-page/old-backend split. Require exclusive ownership.
    allow_reuse_address = os.name != 'nt'
    demo_state: DemoState

    def server_bind(self):
        if os.name == 'nt':
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def server_close(self):
        if not hasattr(self, 'demo_state'):
            super().server_close()
            return
        with self.demo_state.lock:
            sessions = list(self.demo_state.sessions.values())
            workers = list(self.demo_state.workers)
        for session in sessions:
            session.stop()
        for worker in workers:
            # Active inference has a killable worker deadline. Session shutdown
            # preserves results and releases child processes before exiting.
            worker.join(self.demo_state.profile_timeout+5)
        super().server_close()


def make_server(root: Path, *, port=8765, device='cuda', face_model: Path = MODEL_PATH, runner=run_video,
                preprocessing_mode='official',profile_timeout=180,cache_max_bytes=512*1024*1024,
                stream_factory=StreamingSession, stream_policies=None):
    if preprocessing_mode not in ('official','legacy'):raise ValueError('Invalid preprocessing mode')
    state = DemoState(root, device, face_model, runner,preprocessing_mode,profile_timeout,cache_max_bytes,stream_factory,stream_policies)

    class Handler(BaseHTTPRequestHandler):
        def local_request(self):
            port = cast(DemoServer, self.server).server_port
            allowed = {f'127.0.0.1:{port}', f'localhost:{port}'}
            origin = self.headers.get('Origin')
            return self.headers.get('Host') in allowed and (origin is None or origin in {'http://'+h for h in allowed})

        def send_json(self, value, code=200):
            content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(content)

        def send_file(self, path, mime=None):
            if not path.is_file():
                self.send_json(dict(error='檔案尚未產生'), 404)
                return
            for attempt in range(20):
                try:
                    stream = path.open('rb')
                    break
                except PermissionError:
                    if attempt == 19:
                        raise
                    time.sleep(.01)
            with stream:
                self.send_open_file(stream, path, mime)

        def send_open_file(self, stream, path, mime):
            # Header size and content must refer to the same opened version,
            # even when a streaming report is atomically replaced meanwhile.
            size = os.fstat(stream.fileno()).st_size
            start, end = 0, size-1
            requested = self.headers.get('Range')
            if requested:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
                if not match or not any(match.groups()):
                    self.send_json(dict(error='Invalid range'), 416)
                    return
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = min(size-1, int(last)) if last else size-1
                else:
                    start = max(0, size-int(last))
                if start > end or start >= size:
                    self.send_json(dict(error='Invalid range'), 416)
                    return
            self.send_response(206 if requested else 200)
            self.send_header('Content-Type', mime or mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(end-start+1))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if requested:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if self.command == 'HEAD':
                return
            stream.seek(start)
            remaining = end-start+1
            while remaining:
                block = stream.read(min(1024*1024, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            if not self.local_request():
                self.send_json(dict(error='Local requests only'), 403)
                return
            path = urlsplit(self.path).path
            if path == '/':
                self.send_file(PAGE, 'text/html; charset=utf-8')
            elif path == '/favicon.ico':
                self.send_response(204)
                self.send_header('Content-Length', '0')
                self.end_headers()
            elif path == '/api/config':
                self.send_json(dict(api_version=2, analysis_modes=['offline','stream'],
                                    stream_policies=[dict(name=n,policy_id=p['policy_id'],
                                        order=[s['detector'] for s in p['stages']]) for n,p in state.stream_policies.items()],
                                    server_pid=os.getpid(), server_started=state.server_started,
                                    server_build=state.server_build,
                                    device=state.device, max_bytes=VideoLimits().max_bytes,
                                    default_preprocessing_mode=state.preprocessing_mode, profile_timeout=state.profile_timeout, cache_max_bytes=state.cache_max_bytes,
                                    limits=f'MP4 · ≤200 MiB · ≤{VideoLimits().max_seconds} 秒（{VideoLimits().max_seconds // 60} 分鐘） · ≤1920×1080 像素 · 最多 32 幀'))
            elif path == '/api/jobs':
                with state.lock:
                    self.send_json([dict(id=j['id'], filename=j['filename'], phase=j['phase'],analysis_mode=j.get('analysis_mode','offline'))
                                    for j in list(state.jobs.values())[-10:][::-1]])
            else:
                match = re.fullmatch(r'/api/jobs/([0-9a-f]{32})', path)
                file_match = re.fullmatch(r'/jobs/([0-9a-f]{32})/(input\.mp4|report\.json|manifest\.json|windows\.csv|preprocessing\.json|partial_trace\.json|error\.json|decisions\.csv|trace\.csv|frames/\d{8}\.png|profiles/(?:dlib81|retinaface|yunet_legacy)/(?:\d{8}_\d{4}\.png|partial\.json|result\.json|worker_error\.json))', path)
                if match and match[1] in state.jobs:
                    self.send_json(state.snapshot(match[1]))
                elif file_match and file_match[1] in state.jobs:
                    folder = state.root/file_match[1]
                    filename = file_match[2]
                    target=(folder/filename if filename=='input.mp4' else folder/'result'/filename).resolve()
                    if not target.is_relative_to(folder.resolve()):self.send_json(dict(error='Invalid artifact path'),403)
                    else:self.send_file(target)
                else:
                    self.send_json(dict(error='找不到工作或檔案'), 404)

        def do_POST(self):
            if not self.local_request():
                self.send_json(dict(error='Local requests only'), 403)
                return
            parsed = urlsplit(self.path)
            stop_match = re.fullmatch(r'/api/jobs/([0-9a-f]{32})/stop', parsed.path)
            if stop_match:
                code = state.stop(stop_match[1])
                self.send_json(dict(status='stop_requested' if code == 202 else 'not_active'), code)
                return
            if parsed.path != '/api/jobs':
                self.send_json(dict(error='Not found'), 404)
                return
            query = parse_qs(parsed.query)
            filename = query.get('filename', [''])[0].replace('\\', '/').rsplit('/', 1)[-1]
            policy = query.get('policy', ['early'])[0]
            mode=query.get('preprocessing_mode',[state.preprocessing_mode])[0]
            analysis_mode = query.get('analysis_mode', ['offline'])[0]
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                length = 0
            catalog = state.stream_policies if analysis_mode=='stream' else POLICIES
            if not filename.lower().endswith('.mp4') or len(filename) > 180 or policy not in catalog or mode not in ('official','legacy') or \
                    analysis_mode not in ('offline','stream') or \
                    query.get('exploratory') != ['true'] or not 0 < length <= VideoLimits().max_bytes:
                self.send_json(dict(error='需有效 MP4、大小 ≤200 MiB，並確認探索模式'), 400)
                return
            try:
                job_id = state.reserve(filename, policy,mode,analysis_mode)
            except RuntimeError as error:
                self.send_json(dict(error=str(error)), 409)
                return
            except ValueError as error:
                self.send_json(dict(error=str(error)),400)
                return
            try:
                self.connection.settimeout(60)
                with (state.root/job_id/'input.mp4').open('xb') as stream:
                    remaining = length
                    while remaining:
                        block = self.rfile.read(min(1024*1024, remaining))
                        if not block:
                            raise ValueError('上傳提前中斷')
                        stream.write(block)
                        remaining -= len(block)
                state.update(job_id, 'queued', '準備影片處理')
                self.send_json(dict(id=job_id, analysis_mode=analysis_mode, api_version=2), 202)
                state.start(job_id)
            except (OSError, ValueError) as error:
                state.fail(job_id, str(error))
                self.send_json(dict(error=str(error), id=job_id), 400)

    server = DemoServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    server.demo_state = state
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--runs-root', type=Path, default=Path(__file__).resolve().parents[2]/'data/video_demo_runs')
    parser.add_argument('--face-model', type=Path, default=MODEL_PATH)
    parser.add_argument('--open-browser', action='store_true')
    parser.add_argument('--preprocessing-mode',choices=('official','legacy'),default='official')
    parser.add_argument('--profile-timeout',type=float,default=180)
    parser.add_argument('--cache-mib',type=int,default=512)
    parser.add_argument('--stream-policy',action='append',default=[],metavar='NAME=JSON',help='Register a frozen streaming policy in the local UI')
    args = parser.parse_args(argv)
    registered={}
    for item in args.stream_policy:
        name,sep,path=item.partition('=')
        if not sep or not path or name in registered:parser.error('Use unique NAME=JSON policy registrations')
        registered[name]=Path(path)
    server = make_server(args.runs_root, port=args.port, device=args.device, face_model=args.face_model,
                         preprocessing_mode=args.preprocessing_mode,profile_timeout=args.profile_timeout,cache_max_bytes=args.cache_mib*1024*1024,
                         stream_policies=registered)
    address = f'http://127.0.0.1:{server.server_port}'
    print('CASCAde Video + Streaming Demo: '+address, flush=True)
    if args.open_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nDemo stopped. Results preserved.', flush=True)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
