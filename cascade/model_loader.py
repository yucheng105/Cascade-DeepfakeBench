"""Reuse original model classes with complete, strict checkpoint restoration."""
from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
import sys

from .policy import DETECTORS

REPO_ROOT=Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_state(checkpoint: dict, detector: str) -> dict:
    if not isinstance(checkpoint,dict):
        raise ValueError('Checkpoint must be a state dictionary or model wrapper')
    state=checkpoint.get('model',checkpoint.get('state_dict',checkpoint))
    if not isinstance(state,dict) or not state:
        raise ValueError('Checkpoint has no model parameters')
    mapped={}
    for key,value in state.items():
        if not isinstance(key,str):
            raise ValueError('Checkpoint keys must be strings')
        key=key.removeprefix('module.')
        if detector=='sbi':
            if key.startswith('net._fc.'):
                key='backbone.last_layer.'+key[len('net._fc.'):]
            elif key.startswith('net.'):
                key='backbone.efficientnet.'+key[len('net.'):]
        if key in mapped:
            raise ValueError('Checkpoint key mapping collision')
        mapped[key]=value
    return mapped


def load_model(detector: str, config: dict, checkpoint_path: Path, device):
    import torch
    if detector not in DETECTORS or config.get('model_name')!=detector:
        raise ValueError('Model name/config mismatch')
    # Explicit runtime-only registration avoids unrelated detectors' heavy dependencies.
    os.environ['DEEPFAKEBENCH_RUNTIME_ONLY']='1'
    training=str(REPO_ROOT/'training')
    if training not in sys.path:
        sys.path.insert(0,training)
    registry=importlib.import_module('detectors').DETECTOR
    options={**config,'inference_checkpoint_only':True,'pretrained':None}
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    state=checkpoint_state(checkpoint,detector)
    model=registry[detector](options)
    model.load_state_dict(state,strict=True)
    model=model.to(device).eval()
    return model
