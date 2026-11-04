"""Detector-specific normalization over the same selected source frame paths."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import time
from typing import Any

from .model_loader import load_model,sha256
from .video_scoring import VideoScoreAccumulator, VideoScore


class FrameDetectorAdapter:
    def __init__(self, name: str, config_path: Path, checkpoint_path: Path, device,
                 batch_size: int = 16, warmup: bool = True, *,
                 sampling: str = 'first_N_numeric_sorted_frames',
                 input_profile: str = 'preprocessed_face_crops'):
        import yaml
        self.name=name
        self.config=yaml.safe_load(config_path.read_text(encoding='utf-8'))
        if self.config.get('model_name')!=name:
            raise ValueError('Adapter config detector mismatch')
        if batch_size<1:
            raise ValueError('Batch size must be positive')
        self.checkpoint=checkpoint_path.resolve()
        self._resident: Any = None
        self.device,self.batch_size,self.warmup=device,batch_size,warmup
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        interpolation=self.config.get('resize_interpolation','cubic')
        if interpolation not in ('linear','cubic'):
            raise ValueError('Unsupported resize interpolation')
        profile=dict(resolution=self.config['resolution'],mean=self.config['mean'],std=self.config['std'],
                     resize='opencv_INTER_'+interpolation.upper(),color='RGB',aggregation='mean_frame_probability',
                     input=input_profile,sampling=sampling)
        if len(profile['mean'])!=3 or len(profile['std'])!=3 or any(x<=0 for x in profile['std']):
            raise ValueError('Invalid detector normalization')
        self.provenance: dict[str, Any]=dict(detector=name,checkpoint=str(self.checkpoint),checkpoint_sha256=sha256(self.checkpoint),
                             config=str(config_path.resolve()),config_sha256=sha256(config_path),
                             initialization='complete_checkpoint_strict_no_imagenet_preload',
                             checkpoint_mapping='official_net_to_backbone' if name=='sbi' else 'detector_state_dict',
                             backbone_stem='efficientnet_native_same_padding' if name=='sbi' else 'original_xception',
                             preprocessing=profile,preprocessing_sha256=hashlib.sha256(
                                 json.dumps(profile,sort_keys=True).encode()).hexdigest())

    def _image(self, path: str):
        import cv2
        import numpy as np
        import torch
        # Same decode/color/resize as DeepfakeAbstractBaseDataset.load_rgb.
        image=cv2.imread(path)
        if image is None:
            raise ValueError(f'Cannot decode frame: {path}')
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        interpolation=cv2.INTER_LINEAR if self.config.get('resize_interpolation')=='linear' else cv2.INTER_CUBIC
        image=cv2.resize(image,(self.config['resolution'],self.config['resolution']),interpolation=interpolation)
        tensor=torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).float().div_(255.)
        mean=torch.tensor(self.config['mean']).view(3,1,1)
        std=torch.tensor(self.config['std']).view(3,1,1)
        return (tensor-mean)/std

    def open(self):
        """Explicit session lifetime; offline callers retain per-call loading."""
        import torch
        if self._resident is not None:
            return
        device = torch.device(self.device)
        model = load_model(self.name, self.config, self.checkpoint, device)
        with torch.inference_mode():
            if self.warmup:
                dummy = torch.zeros(1, 3, self.config['resolution'], self.config['resolution'], device=device)
                model({'image': dummy, 'label': torch.zeros(1, dtype=torch.long, device=device),
                       'track_metrics': False}, inference=True)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        self._resident = model

    def close(self):
        import torch
        self._resident = None
        if torch.device(self.device).type == 'cuda':
            torch.cuda.empty_cache()

    def infer(self, records: list[dict]):
        import torch
        start=time.perf_counter()
        device=torch.device(self.device)
        def synchronize():
            if device.type=='cuda':torch.cuda.synchronize(device)
        resident = self._resident is not None
        model=self._resident if resident else load_model(self.name,self.config,self.checkpoint,device)
        synchronize()
        load_seconds=time.perf_counter()-start
        warmup_seconds=0.
        forward_seconds=0.
        per_video_forward_ms={r['video_name']:0. for r in records}
        accumulator=VideoScoreAccumulator()
        face_values: dict[str,list[float]]={r['video_name']:[] for r in records}
        try:
            with torch.inference_mode():
                if self.warmup and not resident:
                    began=time.perf_counter()
                    dummy=torch.zeros(1,3,self.config['resolution'],self.config['resolution'],device=device)
                    model({'image':dummy,'label':torch.zeros(1,dtype=torch.long,device=device),
                           'track_metrics':False},inference=True)
                    synchronize();warmup_seconds=time.perf_counter()-began
                frames=[(path,r['label'],r['video_name']) for r in records for path in r['frames']]
                for offset in range(0,len(frames),self.batch_size):
                    batch=frames[offset:offset+self.batch_size]
                    images=torch.stack([self._image(path) for path,_,_ in batch]).to(device)
                    # Model forward requires a tensor; this placeholder is NOT Ground Truth.
                    labels=torch.tensor([0 if label is None else label for _,label,_ in batch],
                                        dtype=torch.long,device=device)
                    synchronize();began=time.perf_counter()
                    output=model({'image':images,'label':labels,'track_metrics':False},inference=True)
                    synchronize();elapsed=time.perf_counter()-began
                    forward_seconds+=elapsed
                    for _,_,video_name in batch:
                        per_video_forward_ms[video_name]+=elapsed*1000/len(batch)
                    if 'prob' not in output or output['prob'].numel()!=len(batch):
                        raise ValueError('Detector output must provide one fake probability per frame')
                    accumulator.update(output['prob'].detach().cpu().reshape(-1).tolist(),
                                       [label for _,label,_ in batch],[name for _,_,name in batch])
                    for value,(_,_,video_name) in zip(output['prob'].detach().cpu().reshape(-1).tolist(),batch):
                        face_values[video_name].append(value)
        finally:
            del model
            if device.type=='cuda' and not resident:torch.cuda.empty_cache()
        scores=accumulator.finalize()
        if any('face_groups' in r for r in records):
            grouped=[]
            for r in records:
                groups=r.get('face_groups')
                if not groups or [p for g in groups for p in g['faces']]!=r['frames']:
                    raise ValueError('Face groups must exactly partition the selected source-frame faces')
                if len({g['frame_index'] for g in groups})!=len(groups):
                    raise ValueError('Source frame IDs must be unique within a profile')
                values=face_values[r['video_name']];offset=0;frame_values=[]
                for group in groups:
                    n=len(group['faces'])
                    if not n:raise ValueError('Empty source-frame face group')
                    frame_values.append(max(values[offset:offset+n]));offset+=n
                grouped.append(VideoScore(r['video_name'],r['label'],sum(frame_values)/len(frame_values),len(frame_values)))
            scores=tuple(grouped)
        return scores,dict(model_load_seconds=load_seconds,warmup_seconds=warmup_seconds,
                                           forward_seconds=forward_seconds,stage_seconds=time.perf_counter()-start,
                                           frame_count=sum(s.frame_count for s in scores),face_count=len(frames),batch_size=self.batch_size,
                                           per_video_forward_ms=per_video_forward_ms)
