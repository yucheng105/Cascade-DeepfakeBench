"""Shared lossless raw samples, lazy profile artifacts, and interruptible workers."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2

from .model_loader import REPO_ROOT,sha256
from .video_input import VideoLimits,prepare_video,MODEL_PATH

ASSET_LOCK=REPO_ROOT/'cascade/assets/preprocessing/assets.json'
PROFILE_FOR_MODEL={'xception':'dlib81','ucf':'dlib81','sbi':'retinaface'}


def atomic_json(path,value,*,indent=2):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=indent,allow_nan=False)+'\n',encoding='utf-8')
    # Windows scanners/readers can briefly hold a handle without FILE_SHARE_DELETE.
    for attempt in range(20):
        try:
            os.replace(temporary,path)
            break
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)


def load_assets(lock=ASSET_LOCK):
    if not lock.is_file():raise ValueError('Official preprocessing setup required: python -m cascade.prepare_preprocessing')
    data=json.loads(lock.read_text(encoding='utf-8'))
    for asset in data['assets'].values():
        path=(REPO_ROOT/asset['relative_path']).resolve()
        if not path.is_file() or sha256(path)!=asset['sha256']:raise ValueError('Preprocessing asset missing/hash mismatch: '+str(path))
        asset['path']=str(path)
    for name,version in data['packages'].items():
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:raise ValueError('Official preprocessing setup required: '+name) from error
        if actual!=version:raise ValueError(f'Preprocessing package mismatch: {name} expected {version}, found {actual}')
    return data


def preflight(mode,names,face_model=MODEL_PATH):
    if mode not in ('official','legacy'):raise ValueError('Unknown preprocessing mode')
    profiles=set(PROFILE_FOR_MODEL[n] for n in names) if mode=='official' else {'yunet_legacy'}
    if mode=='official':
        lock=load_assets();assets=lock['assets'];packages=lock['packages']
    else:
        from .download_face_model import MODEL_SHA256
        if not face_model.is_file() or sha256(face_model)!=MODEL_SHA256:raise ValueError('YuNet asset missing/hash mismatch')
        assets={'yunet':dict(path=str(face_model.resolve()),sha256=MODEL_SHA256)};packages={}
    result={}
    for profile in profiles:
        settings=(dict(detector='dlib_frontal',upsample=1,landmark_indices=[37,44,30,49,55],
                       scale=1.3,post_alignment_check=True,resolution=256,
                       reference='DeepfakeBench@f188b1c105465e2e5377eb536a95022ae0e4522d') if profile=='dlib81'
                  else dict(detector='retinaface_resnet50_2020-07-20',max_size=2048,confidence=.7,nms=.4,
                            margin_per_side=.125,min_area_ratio=.5,resolution=380,aggregation='max_face_then_mean_frame',
                            reference='SelfBlendedImages@d6d9a351f32334ae9a3a250872c05836e0dd0f2c') if profile=='retinaface'
                  else dict(detector='yunet2023mar',max_detection_side=640,resolution=256,scale=1.3,largest_face=True))
        relevant=(['predictor'] if profile=='dlib81' else ['retinaface'] if profile=='retinaface' else ['yunet'])
        definition=dict(id=profile,settings=settings,packages=packages,
                        assets={k:assets[k] for k in relevant},
                        implementation_sha256=sha256(REPO_ROOT/'cascade/face_profiles.py'),
                        worker_sha256=sha256(Path(__file__)))
        definition['profile_sha256']=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()
        result[profile]=definition
    return result


def run_process(request,timeout):
    """A separate process lets timeouts stop native decoder/detector calls."""
    folder=Path(request['output']);folder.mkdir(parents=True,exist_ok=True)
    req=folder/'request.json';atomic_json(req,request)
    log=folder/'worker.log'
    began=time.monotonic()
    with log.open('wb') as stream:
        process=subprocess.Popen([sys.executable,'-B','-m','cascade.preprocessing','--request',str(req)],
            cwd=REPO_ROOT,stdout=stream,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            process.kill();process.wait()
            raise ValueError('preprocessing timeout; worker terminated') from error
    if process.returncode:
        error_file=folder/'worker_error.json'
        message=json.loads(error_file.read_text(encoding='utf-8'))['error'] if error_file.is_file() else 'worker failed; see worker.log'
        raise ValueError(message)
    result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    result['seconds']=time.monotonic()-began
    return result


def execute_request(request,face_factory=None):
    output=Path(request['output']);output.mkdir(parents=True,exist_ok=True)
    budget=request['cache_remaining_bytes']
    if request['kind']=='raw':
        class Raw:
            def crop(self,image):return image,{}
        paths,summary=prepare_video(Path(request['video']),output,VideoLimits(**request['limits']),Raw(),cache_max_bytes=budget)
        for key in ('valid_face_frames','face_selection','face_detector_threshold','face_detector_nms'):summary.pop(key,None)
        summary['profile']='raw_bgr_lossless_v1'
        summary['cache_bytes']=sum(Path(p).stat().st_size for p in paths)
        return summary
    from .face_profiles import build_faces
    profile=request['profile']
    detector=(face_factory(profile) if face_factory else build_faces(profile,request['assets'],request['device']))
    groups=[];samples=[];cache_bytes=0
    for sample in request['samples']:
        image=cv2.imread(sample['path'])
        if image is None or sha256(Path(sample['path']))!=sample['sha256']:raise ValueError('Raw sample missing/hash mismatch')
        faces=detector(image)
        mapped=[]
        for face_index,(crop,details) in enumerate(faces):
            path=output/f"{sample['frame_index']:08d}_{face_index:04d}.png"
            ok,encoded=cv2.imencode('.png',crop)
            if not ok:raise ValueError('Face encoding failed')
            if cache_bytes+encoded.nbytes>budget:raise ValueError('cache budget exceeded')
            with path.open('xb') as stream:stream.write(encoded.tobytes())
            cache_bytes+=encoded.nbytes
            mapped.append(dict(path=str(path.resolve()),sha256=sha256(path),**details))
        entry=dict(frame_index=sample['frame_index'],timestamp_seconds=sample['timestamp_seconds'],
                   faces=mapped,face_count=len(mapped))
        if mapped:groups.append(dict(frame_index=entry['frame_index'],faces=[f['path'] for f in mapped]))
        else:entry['reason']=getattr(detector,'last_rejection_reason','no_valid_face')
        samples.append(entry)
        atomic_json(output/'partial.json',dict(profile=profile,samples=samples,cache_bytes=cache_bytes))
    return dict(profile=profile,samples=samples,face_groups=groups,valid_face_frames=len(groups),
                face_count=sum(len(g['faces']) for g in groups),cache_bytes=cache_bytes)


class ProfileCache:
    def __init__(self,video,output,mode,names,device,limits,timeout,cache_bytes,face_model=MODEL_PATH,progress=None):
        if not math.isfinite(timeout) or timeout<=0 or type(cache_bytes) is not int or cache_bytes<1:
            raise ValueError('Invalid profile timeout/cache budget')
        self.output=output;self.mode=mode;self.device=device;self.timeout=timeout
        self.limit=cache_bytes;self.progress=progress
        definitions=preflight(mode,names,face_model)
        self.profiles={name:dict(state='not_reached',definition=definition) for name,definition in definitions.items()}
        self.mapping={n:PROFILE_FOR_MODEL[n] if mode=='official' else 'yunet_legacy' for n in names}
        self.raw=run_process(dict(kind='raw',video=str(video.resolve()),output=str(output/'raw'),
                                 limits=asdict(limits),cache_remaining_bytes=self.limit),limits.decode_timeout_seconds)
        self.used_bytes=self.raw['cache_bytes']
        identity=[(s['frame_index'],s['sha256']) for s in self.raw['samples']]
        for state in self.profiles.values():
            state['cache_key']=hashlib.sha256(json.dumps(dict(
                source=sha256(video),samples=identity,definition=state['definition']),sort_keys=True).encode()).hexdigest()
        self.save()

    def summary(self):
        return dict(mode=self.mode,sampling=self.raw['sampling'],duration_seconds=self.raw['duration_seconds'],
                    sampled_frame_count=self.raw['sampled_frame_count'],raw=self.raw,profiles=self.profiles,
                    model_profiles=self.mapping,cache_bytes=self.used_bytes,cache_limit_bytes=self.limit,
                    profile_timeout_seconds=self.timeout)

    def save(self):atomic_json(self.output/'preprocessing.json',self.summary())

    def get(self,detector):
        name=self.mapping[detector];state=self.profiles[name]
        if state['state']=='used':return state['result']
        state['state']='running';self.save()
        if self.progress:self.progress('preprocessing','人臉前處理：'+name)
        try:
            assets=state['definition'].get('assets',{})
            result=run_process(dict(kind='faces',profile=name,assets=assets,device=self.device,
                output=str(self.output/'profiles'/name),samples=self.raw['samples'],
                cache_remaining_bytes=self.limit-self.used_bytes),self.timeout)
            self.used_bytes+=result['cache_bytes']
            state['result']=result
            if not result['valid_face_frames']:raise ValueError('Reached profile has no valid faces: '+name)
            state.update(state='used',result=result)
            self.save();return result
        except Exception as error:
            state.update(state='failed',error=str(error));self.save();raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_text(encoding='utf-8'))
    folder=Path(request['output'])
    try:atomic_json(folder/'result.json',execute_request(request))
    except Exception as error:
        atomic_json(folder/'worker_error.json',dict(error=str(error)));raise
