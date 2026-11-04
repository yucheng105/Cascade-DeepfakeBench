"""Admitted event timeline and shared, bounded sliding-window selection.

v2 ticks are anchored at source time zero. At a committed progress watermark,
select the newest admitted frame at/before each tick. A source identity is
sampled once globally, then shared by reference across overlapping windows.
"""
from __future__ import annotations

import hashlib
import math
import queue
import threading
import time
from typing import Any, TypedDict, cast


class WindowIdentity(TypedDict):
    window_contract_version: int
    source_id: str
    session_id: str
    window_id: str
    index: int
    start_seconds: float
    end_seconds: float
    nominal_end_seconds: float
    observed_end_seconds: float
    tail: bool


class WindowInput(WindowIdentity):
    samples: list[dict[str, Any]]


class WindowResult(WindowIdentity, total=False):
    """Serialized result inherits input identity; pixels never enter reports."""
    source_frames: list[dict[str, Any]]
    planned_samples: int
    outcome: str
    decision: str | None
    score: float | None
    exit_stage: int | None
    trace: list[dict[str, Any]]
    profiles: dict[str, Any]
    policy_sha256: str
    input_contract: dict[str, Any]
    reason: str


def positive(value, field):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(field+' must be finite and positive')
    return float(value)


def frame_evidence(sample):
    return {k:sample[k] for k in ('frame_id','frame_index','timestamp_seconds','timestamp_basis',
                                 'sampling_tick_seconds','raw_pixels_sha256','admission_event_index') if k in sample}


def window_record(window: WindowInput) -> WindowResult:
    return cast(WindowResult,{**{k:v for k,v in window.items() if k!='samples'},
            'source_frames':[frame_evidence(s) for s in window['samples']],
            'planned_samples':len(window['samples']), 'outcome':'collecting','decision':None})


class WindowTimeline:
    def __init__(self, source_id, session_id, *, window_seconds=4., stride_seconds=4., sample_hz=2.,
                 cache_max_bytes=256*1024*1024, max_windows=10000, max_seconds=3600., max_events=200000):
        self.duration=positive(window_seconds,'window_seconds')
        self.stride=positive(stride_seconds,'stride_seconds')
        self.hz=positive(sample_hz,'sample_hz')
        positive(cache_max_bytes,'cache_max_bytes')
        positive(max_seconds,'max_seconds')
        if any(type(v) is not int or v<1 for v in (max_windows,max_events)):
            raise ValueError('Record budgets must be positive integers')
        if self.stride > self.duration or math.ceil(self.duration/self.stride)>128 or self.duration*self.hz>10000:
            raise ValueError('Unsupported stride or active window capacity')
        self.source_id,self.session_id=source_id,session_id
        self.budget,self.max_windows,self.max_seconds,self.max_events=cache_max_bytes,max_windows,max_seconds,max_events
        self.watermark=0.
        self.next_tick=0
        self.next_window=0
        self.candidates=[]
        self.samples=[]
        self.identities={}
        self.frame_indices={}
        self.published_samples=0
        self.last_selected=None
        self.events=0
        self.diagnostics=[]
        self.closed=False
        self.peak_raw_bytes=0

    def log(self, reason, **fields):
        if len(self.diagnostics)>=1000:
            raise ValueError('Source diagnostic budget exceeded')
        self.diagnostics.append(dict(reason=reason,**fields))

    def accept(self,event):
        if self.closed:
            raise ValueError('Source already ended')
        self.events+=1
        if self.events>self.max_events:
            raise ValueError('Source event budget exceeded')
        kind=event.get('type')
        if kind=='error':
            raise ValueError('Source error: '+str(event.get('message','unspecified')))
        if kind=='frame':
            timestamp=event.get('timestamp_seconds')
            tick=event.get('sampling_tick_seconds',timestamp)
            if any(isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or t<0 for t in (timestamp,tick)):
                raise ValueError('Frame requires finite source time')
            if timestamp>tick or not isinstance(event.get('timestamp_basis'),str) or not 0<len(event['timestamp_basis'])<=128:
                raise ValueError('Frame timestamp basis missing or future frame')
            identity=event.get('frame_id')
            if not isinstance(identity,str) or not identity or len(identity)>256:
                raise ValueError('Invalid frame identity')
            image=event['image']
            if len(image.shape)!=3 or image.shape[2]!=3 or str(image.dtype)!='uint8':
                raise ValueError('Frame must be uint8 BGR HWC')
            digest=hashlib.sha256(image.tobytes()).hexdigest()
            signature=(digest,timestamp,event['timestamp_basis'],event.get('frame_index'))
            if identity in self.identities:
                if self.identities[identity]!=signature:
                    self.log('conflicting_frame_identity',frame_id=identity)
                    raise ValueError('Conflicting frame identity: '+identity)
                self.log('duplicate_frame',frame_id=identity)
                return []
            if len(self.identities)>=self.max_events:
                raise ValueError('Frame identity budget exceeded')
            self.identities[identity]=signature
            if tick < self.watermark:
                self.log('late_frame_rejected',frame_id=identity,seconds=tick)
                return []
            if self.candidates and tick < self.candidates[-1]['admission_seconds']:
                raise ValueError('Frame admission time regressed before watermark')
            index=event.get('frame_index',len(self.identities)-1)
            if type(index) is not int or index<0:
                raise ValueError('Invalid frame_index')
            if index in self.frame_indices and self.frame_indices[index]!=identity:
                raise ValueError('Conflicting source frame_index')
            self.frame_indices[index]=identity
            self.candidates.append(dict(frame_id=identity,frame_index=index,timestamp_seconds=timestamp,
                timestamp_basis=event['timestamp_basis'],raw_pixels_sha256=digest,image=image,
                admission_seconds=tick,admission_event_index=self.events))
            self.check_budget()
            return []
        if kind not in ('progress','end'):
            raise ValueError('Unsupported source event')
        horizon=event.get('seconds')
        if isinstance(horizon,bool) or not isinstance(horizon,(int,float)) or not math.isfinite(horizon) or horizon<self.watermark:
            raise ValueError('Invalid or regressed source watermark')
        if horizon>self.max_seconds:
            raise ValueError('Source duration budget exceeded')
        if math.ceil(horizon*self.hz)>self.max_events:
            raise ValueError('Sampling tick budget exceeded')
        while self.next_tick/self.hz < horizon-1e-10:
            tick=self.next_tick/self.hz
            eligible=[s for s in self.candidates if s['admission_seconds']<=tick+1e-10]
            if eligible:
                latest=eligible[-1]
                if latest['frame_id']!=self.last_selected:
                    self.samples.append(dict(latest,sampling_tick_seconds=tick))
                    self.last_selected=latest['frame_id']
                self.candidates=[s for s in self.candidates if s['admission_seconds']>tick]+[latest]
                self.candidates.sort(key=lambda s:s['admission_seconds'])
            self.next_tick+=1
            self.check_budget()
        self.watermark=horizon
        windows=[]
        while True:
            start=self.next_window*self.stride
            if start>=horizon-1e-10 or (start+self.duration>horizon+1e-10 and kind!='end'):
                break
            if self.next_window>=self.max_windows:
                raise ValueError('Window record budget exceeded')
            end=min(start+self.duration,horizon)
            samples=[s for s in self.samples if start-1e-10<=s['sampling_tick_seconds']<end-1e-10]
            self.published_samples+=len(samples)
            if self.published_samples>self.max_events:
                raise ValueError('Window sample metadata budget exceeded')
            windows.append(dict(window_contract_version=2,source_id=self.source_id,session_id=self.session_id,
                window_id=f'{self.session_id}:{self.next_window}',index=self.next_window,start_seconds=start,
                end_seconds=end,nominal_end_seconds=start+self.duration,observed_end_seconds=end,
                tail=end<start+self.duration-1e-10,samples=samples))
            self.next_window+=1
        self.samples=[s for s in self.samples if s['sampling_tick_seconds']>=self.next_window*self.stride-1e-10]
        self.closed=kind=='end'
        return windows

    def check_budget(self):
        unique={id(s['image']):s['image'] for s in self.samples+self.candidates}
        used=sum(image.nbytes for image in unique.values())
        self.peak_raw_bytes=max(self.peak_raw_bytes,used)
        if used>self.budget:
            raise ValueError('Source raw cache budget exceeded')


# At most one uncooperative injected reader can remain alive in the process.
# MP4 native reads retain the killable process boundary in streaming.VideoSource.
_reader_capacity=threading.BoundedSemaphore(1)


class EventReader:
    """One request at a time, bounded wait; source must honor cancellation.

    A misbehaving library source is quarantined and prevents a replacement
    reader until it returns, rather than leaking a new thread per timeout.
    """
    def __init__(self, source, stop_event, timeout):
        if not _reader_capacity.acquire(blocking=False):
            raise ValueError('Event reader capacity exhausted by an unclosed source')
        self.source,self.stop,self.timeout=source,stop_event,timeout
        self.requests=queue.Queue(maxsize=1)
        self.responses=queue.Queue(maxsize=1)
        self.closed=threading.Event()
        def run():
            try:
                while not self.closed.is_set():
                    try:self.requests.get(timeout=.05)
                    except queue.Empty:continue
                    if self.closed.is_set():break
                    try:value=(True,source.read_event(timeout=timeout,stop_event=stop_event))
                    except Exception as error:value=(False,error)
                    self.responses.put(value)
            finally:
                try:source.close()
                finally:_reader_capacity.release()
        self.thread=threading.Thread(target=run,daemon=True)
        self.thread.start()

    def read(self):
        self.requests.put_nowait(True)
        deadline=time.monotonic()+self.timeout
        while not self.stop.is_set():
            remaining=deadline-time.monotonic()
            if remaining<=0:
                raise TimeoutError('Event source read timed out')
            try:ok,value=self.responses.get(timeout=min(.05,remaining))
            except queue.Empty:continue
            if self.stop.is_set():
                break
            if not ok:raise value
            return value
        from .streaming import StreamStopped
        raise StreamStopped()

    def close(self):
        self.closed.set()
        self.thread.join(.1)


def mp4_events(source, sample_hz):
    """Legacy floor(frame-index/FPS) fallback; sampling tick is separate."""
    info=source.info
    duration=info['duration_seconds']
    previous=-1
    for number in range(math.ceil(duration*sample_hz-1e-10)):
        tick=number/sample_hz
        index=min(int(tick*info['fps']),info['frame_count']-1)
        if index!=previous:
            yield dict(type='frame',frame_id=str(index),frame_index=index,image=source.read(index),
                timestamp_seconds=index/info['fps'],sampling_tick_seconds=tick,timestamp_basis='frame_index_over_fps')
            previous=index
        yield dict(type='progress',seconds=min((number+1)/sample_hz,duration))
    source.finish()
    yield dict(type='end',seconds=duration)
