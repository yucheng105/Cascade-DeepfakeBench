"""Explicit, offline-verifiable setup for the official face profiles.

Install the documented isolated environment first. Only --download uses network.
The existing 81-point predictor is verified before any replacement is considered.
"""
import argparse
import importlib.metadata
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

from .model_loader import REPO_ROOT,sha256
from .preprocessing import ASSET_LOCK,atomic_json,preflight

EXPECTED={
    'predictor':dict(relative_path='preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat',
        sha256='8cae4375589dd915d9a0a881101bed1bbb4e9887e35e63b024388f1ca25ff869',
        url='https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/shape_predictor_81_face_landmarks.dat'),
    'retinaface':dict(relative_path='cascade/assets/preprocessing/retinaface_resnet50_2020-07-20.pth',
        sha256='b9fdd58c474d809b67166130b42d5913f00b48f912a3491565bc56b8a8a796f2',
        archive_sha256='f168fae3ce6d6d3af69dd712049bfe857d3bc1422f715ac8ddfdf552ed7070ef',
        archive_member='retinaface_resnet50_2020-07-20.pth',
        url='https://github.com/ternaus/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20-f168fae3c.zip')}
PACKAGES={'torch':'2.7.1+cu128','torchvision':'0.22.1+cu128','numpy':'1.26.4',
    'opencv-python-headless':'4.11.0.86','dlib-bin':'20.0.1.post1','scikit-image':'0.24.0',
    'imutils':'0.5.4','retinaface-pytorch':'0.0.8+torch27','albumentations':'1.0.0'}


def prepare(download=False,device='cuda',probe=True):
    for name,version in PACKAGES.items():
        if importlib.metadata.version(name)!=version:raise ValueError('Pinned package required: '+name+'=='+version)
    for asset in EXPECTED.values():
        path=REPO_ROOT/asset['relative_path']
        if not path.exists():
            if not download:raise ValueError('Missing asset; use --download: '+str(path))
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_suffix('.download')
            try:
                with urlopen(asset['url'],timeout=60) as response,temp.open('xb') as stream:
                    size=0
                    while block:=response.read(1024*1024):
                        size+=len(block)
                        if size>150*1024*1024:raise ValueError('Asset exceeds bounded download size')
                        stream.write(block)
                if sha256(temp)!=asset.get('archive_sha256',asset['sha256']):raise ValueError('Downloaded asset hash mismatch')
                if 'archive_member' in asset:
                    with zipfile.ZipFile(temp) as archive:
                        info=archive.getinfo(asset['archive_member'])
                        if info.file_size>150*1024*1024:raise ValueError('Unpacked asset too large')
                        payload=archive.read(info)
                    temp.write_bytes(payload)
                    if sha256(temp)!=asset['sha256']:raise ValueError('Unpacked asset hash mismatch')
                temp.replace(path)
            finally:
                if temp.exists():temp.unlink()
        if sha256(path)!=asset['sha256']:raise ValueError('Existing asset hash mismatch; not overwritten: '+str(path))
    atomic_json(ASSET_LOCK,dict(schema_version=1,assets=EXPECTED,packages=PACKAGES,
        dlib_distribution='dlib-bin, upstream dlib binary build by alesanfra/dlib-wheels; not an official PyPI dlib wheel',
        retinaface_compatibility='0.0.8+torch27 changes only wheel version/dependency metadata; every upstream Python source byte is unchanged; reference parity is required'))
    definitions=preflight('official',['xception','ucf','sbi'])
    if probe:
        import numpy as np
        from .face_profiles import build_faces
        import torch
        torch.set_num_threads(4)
        for profile in ('dlib81','retinaface'):
            faces=build_faces(profile,definitions[profile]['assets'],device)
            result=faces(np.zeros((90,160,3),dtype=np.uint8))
            print(profile+': loaded and executed; blank-image faces='+str(len(result)),flush=True)
            del faces
            if device=='cuda':torch.cuda.empty_cache()
    return definitions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--check-only',action='store_true',help='Verify existing setup without loading detectors or changing assets')
    args=parser.parse_args()
    result=preflight('official',['xception','ucf','sbi']) if args.check_only else prepare(args.download,args.device)
    if not args.check_only:
        atomic_json(REPO_ROOT/'preprocessing-readiness.json',dict(ready=True,profiles=result))
    print('Official preprocessing checks passed; inference requires no downloads.')


if __name__=='__main__':main()
