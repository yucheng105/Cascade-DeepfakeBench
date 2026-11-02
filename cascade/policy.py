"""Versioned Phase 1 -> runtime policy contract, independent of model imports."""
from __future__ import annotations

import copy
import math
import re

DETECTORS = ('xception', 'ucf', 'sbi')
DATASETS = {'FaceForensics++':'ffpp','Celeb-DF-v2':'celeb_df_v2','DFDC':'dfdc'}


def validate_policy(value: dict, dataset_id: str, *, allow_exploratory: bool = False) -> dict:
    if not isinstance(value,dict):
        raise ValueError('Policy must be a JSON object')
    if type(value.get('schema_version')) is not int or value.get('schema_version') != 1 or value.get('kind') != 'sequential':
        raise ValueError('Only sequential policy schema_version=1 is supported')
    if value.get('metric_definition') != 'fake_positive_far_is_fnr' or \
            value.get('comparators') != 'intermediate_real_le_fake_ge_final_fake_ge':
        raise ValueError('Unsupported policy metrics/comparators')
    if not isinstance(value.get('policy_id'),str) or not value['policy_id'].strip():
        raise ValueError('Policy must have a nonempty policy_id')
    if dataset_id not in (*DATASETS.values(), 'uploaded_video') or value.get('dataset_id') != dataset_id:
        raise ValueError('Policy dataset mismatch')
    if value.get('weights') != []:
        raise ValueError('Sequential policies cannot have fusion weights')
    stages=value.get('stages')
    if not isinstance(stages,list) or not 1<=len(stages)<=3:
        raise ValueError('Policy requires 1..3 stages')
    names=[]
    for n,stage in enumerate(stages):
        if not isinstance(stage,dict) or stage.get('detector') not in DETECTORS:
            raise ValueError('Unknown detector')
        names.append(stage['detector'])
        for key in ('low','high'):
            number=stage.get(key)
            if isinstance(number,bool) or not isinstance(number,(int,float)) or \
                    not math.isfinite(number) or not 0<=number<=1:
                raise ValueError('Thresholds must be finite numbers in [0,1]')
        if n==len(stages)-1:
            if stage['low'] != stage['high']:
                raise ValueError('Final stage must have a single threshold')
        elif not stage['low'] < stage['high']:
            raise ValueError('Intermediate stages require low < high')
    if len(set(names))!=len(names):
        raise ValueError('Duplicate policy detectors')
    if not allow_exploratory:
        if value.get('deployment_ready') is not True:
            raise ValueError('Exploratory policy requires --allow-exploratory')
        for field in ('checkpoint_sha256','config_sha256'):
            expected=value.get(field)
            if not isinstance(expected,dict) or any(not isinstance(expected.get(name),str) or
                    re.fullmatch('[0-9a-f]{64}',expected[name]) is None for name in names):
                raise ValueError('Deployment policy requires checkpoint/config SHA-256 bindings')
    return copy.deepcopy(value)
