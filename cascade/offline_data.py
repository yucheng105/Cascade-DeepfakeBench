"""Strict video identity and path-preserving read of existing DeepfakeBench JSON."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from .policy import DATASETS

LABELS={'FF-real':0,'FF-DF':1,'FF-F2F':1,'FF-FS':1,'FF-NT':1,
        'CelebDFv2_real':0,'CelebDFv2_fake':1,'DFDC_Real':0,'DFDC_Fake':1}


def load_records(json_path: Path, rgb_root: Path, dataset: str, *, compression: str = 'c23',
                 max_frames: int = 32) -> list[dict]:
    if dataset not in DATASETS or max_frames<1:
        raise ValueError('Unsupported dataset or nonpositive max_frames')
    root=rgb_root.resolve()
    node=json.loads(json_path.read_text(encoding='utf-8-sig'))[dataset]
    records=[]
    for category,cohorts in sorted(node.items()):
        if category not in LABELS:
            raise ValueError(f'Unknown source label: {category}')
        cohort=cohorts.get('test',{})
        if dataset=='FaceForensics++':
            cohort=cohort.get(compression,{})
        for video_id,info in sorted(cohort.items()):
            if info['label']!=category or not isinstance(video_id,str) or not video_id:
                raise ValueError('Dataset label/identity conflict')
            paths=[]
            # Preserve the current exporter's first-N behavior explicitly. No new sampling claim.
            frames=sorted(info['frames'],key=lambda p:int(PurePosixPath(p.replace('\\','/')).stem))[:max_frames]
            if not frames:
                raise ValueError('Video has no frames')
            for frame in frames:
                relative=PurePosixPath(frame.replace('\\','/'))
                if relative.is_absolute() or '..' in relative.parts or ':' in str(relative):
                    raise ValueError('Unsafe frame path')
                if not relative.parts or relative.parts[0]!=dataset:
                    raise ValueError('Frame dataset prefix conflicts with selected dataset')
                target=(root/Path(*relative.parts)).resolve()
                if not target.is_relative_to(root):
                    raise ValueError('Frame path escapes RGB root')
                if not target.is_file():
                    raise FileNotFoundError(target)
                paths.append(str(target))
            if len(set(paths))!=len(paths):
                raise ValueError('Duplicate frames in one video')
            records.append(dict(video_name=info['label']+'_'+video_id,source_video_id=video_id,
                                source_label=category,label=LABELS[category],frames=paths))
    if not records or len({r['video_name'] for r in records})!=len(records):
        raise ValueError('Dataset cohort must be nonempty with unique video names')
    return records
