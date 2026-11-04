"""Versioned Phase 1 -> runtime policy contract, independent of model imports."""
from __future__ import annotations

import copy
import math
import re

DETECTORS = ('xception', 'ucf', 'sbi')
DATASETS = {'FaceForensics++':'ffpp','Celeb-DF-v2':'celeb_df_v2','DFDC':'dfdc'}
HASH_BINDINGS = ('checkpoint_sha256', 'config_sha256', 'preprocessing_sha256')
LATENCY_MODELS = ('sum_reached_csv_ms_serial', 'sum_reached_forward_ms_serial', 'measured_end_to_end_ms')
VIDEO_SAMPLING = ('first_N_numeric_sorted_frames', 'uniform_unique_frame_indices_v1')
FACE_PROFILES = ('preprocessed_face_crops', 'dlib81', 'retinaface', 'yunet_legacy')
AGGREGATIONS = ('mean_frame_probability', 'max_face_then_mean_frame')


def execution_contract(names, *, unit='video', mode='official', max_frames=32,
                       window_seconds=4., sample_hz=2., stride_seconds=None, window_contract_version=1) -> dict:
    profiles = {'xception':'dlib81','ucf':'dlib81','sbi':'retinaface'}
    value = dict(unit=unit, profiles={n:('preprocessed_face_crops' if mode=='crops' else
        profiles[n] if mode=='official' else 'yunet_legacy') for n in names},
        aggregation={n:'mean_frame_probability' if mode=='crops' else 'max_face_then_mean_frame' for n in names})
    if unit == 'video':
        value.update(sampling='first_N_numeric_sorted_frames' if mode=='crops' else 'uniform_unique_frame_indices_v1',max_frames=max_frames)
    else:
        value.update(sampling='paced_media_time_no_future_frames',window_seconds=window_seconds,
                     stride_seconds=window_seconds,sample_hz=sample_hz)
    if unit == 'window' and window_contract_version == 2:
        value.update(window_contract_version=2, sampling='admitted_tick_grid_v2',
                     stride_seconds=window_seconds if stride_seconds is None else stride_seconds)
    elif window_contract_version != 1 or (stride_seconds is not None and stride_seconds != window_seconds):
        raise ValueError('New cadence requires window_contract_version=2')
    _validate_input_contract(value,list(names))
    return value


def frozen_policy(value=None, *, path=None) -> tuple[dict, str]:
    import hashlib
    import json
    from .video import POLICIES
    if value is not None and path is not None:
        raise ValueError('Choose policy content or policy_path, not both')
    if value is None:
        value=json.loads((path or POLICIES['early']).read_text(encoding='utf-8-sig'))
    result=validate_policy(value,'uploaded_video',allow_exploratory=True)
    return result,hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _finite_number(value, field: str, *, probability: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{field} must be a finite number')
    # Comparing huge JSON integers avoids float conversion overflow in isfinite.
    if probability:
        valid = 0 <= value <= 1
    else:
        valid = value > 0
    if not valid or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError(f'{field} must be finite and '+('in [0,1]' if probability else 'positive'))


def _validate_input_contract(value, names: list[str]) -> None:
    field = 'input_contract'
    if not isinstance(value, dict) or value.get('unit') not in ('video', 'window'):
        raise ValueError(f'{field}.unit must be video or window')
    common = {'unit', 'sampling', 'profiles', 'aggregation'}
    required = common | ({'max_frames'} if value['unit'] == 'video'
                         else {'window_seconds', 'stride_seconds', 'sample_hz'})
    version=value.get('window_contract_version',1)
    if 'window_contract_version' in value:
        if value['unit']!='window' or type(version) is not int or version!=2:
            raise ValueError('Unsupported window_contract_version')
        required |= {'window_contract_version'}
    if set(value) != required:
        raise ValueError(f'{field} requires exactly {sorted(required)}')
    for key, allowed in (('profiles', FACE_PROFILES), ('aggregation', AGGREGATIONS)):
        mapping = value[key]
        if not isinstance(mapping, dict) or set(mapping) != set(names) or any(
                not isinstance(v, str) or v not in allowed for v in mapping.values()):
            raise ValueError(f'{field}.{key} must describe every active detector with a supported value')
    if value['unit'] == 'video':
        if value['sampling'] not in VIDEO_SAMPLING:
            raise ValueError(f'{field}.sampling is unsupported for video')
        if type(value['max_frames']) is not int or value['max_frames'] < 1:
            raise ValueError(f'{field}.max_frames must be a positive integer')
    else:
        if value['sampling'] != ('admitted_tick_grid_v2' if version==2 else 'paced_media_time_no_future_frames'):
            raise ValueError(f'{field}.sampling is unsupported for window')
        for key in ('window_seconds', 'stride_seconds', 'sample_hz'):
            _finite_number(value[key], f'{field}.{key}')
        if value['stride_seconds']>value['window_seconds']:
            raise ValueError('stride_seconds exceeds window_seconds')


def _validate_metadata(value: dict, names: list[str]) -> None:
    if 'deployment_ready' in value and type(value['deployment_ready']) is not bool:
        raise ValueError('deployment_ready must be a boolean')
    formal = value.get('deployment_ready') is True
    if formal and (value.get('research_only') is True or value.get('estimator', 'sample_replay') != 'sample_replay'):
        raise ValueError('Research estimators cannot declare deployment_ready')
    if formal and (value.get('promotion_status') == 'requires_independent_review' or
                   'unverified' in str(value.get('split_status','')).lower() or
                   (isinstance(value.get('validation'),dict) and value['validation'].get('evaluation_meets_target') is False)):
        raise ValueError('Independent validation incomplete; cannot declare deployment_ready')
    required = ('target_far', 'latency_model', 'input_contract', *HASH_BINDINGS)
    if formal:
        for field in required:
            if field not in value:
                raise ValueError('Deployment policy requires '+field)
    if 'target_far' in value:
        _finite_number(value['target_far'], 'target_far', probability=True)
    if 'latency_model' in value and value['latency_model'] not in LATENCY_MODELS:
        raise ValueError('latency_model must name a supported cost scope')
    if 'objective' in value and value['objective'] not in ('accuracy', 'latency'):
        raise ValueError('objective must be accuracy or latency')
    for field in HASH_BINDINGS:
        if field not in value:
            continue
        mapping = value[field]
        if not isinstance(mapping, dict) or set(mapping) != set(names) or any(
                not isinstance(v, str) or re.fullmatch('[0-9a-f]{64}', v) is None
                for v in mapping.values()):
            raise ValueError(f'{field} requires a SHA-256 binding for every active detector')
    if 'input_contract' in value:
        _validate_input_contract(value['input_contract'], names)


def validate_runtime_input(policy: dict, actual: dict) -> None:
    """Compare a validated policy's declared unit/cadence/geometry before work starts."""
    expected = policy.get('input_contract')
    if expected is not None and expected != actual:
        fields = sorted(k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k))
        raise ValueError('Policy input_contract mismatch: '+', '.join(fields))


def validate_runtime_bindings(policy: dict, models: dict) -> None:
    """Entry-point deployment gate; the pure controller cannot inspect actual assets.

    Declared bindings are enforced even with --allow-exploratory. A binding only
    establishes compatibility, never statistical calibration or a FAR guarantee.
    """
    for field in HASH_BINDINGS:
        for name, expected in policy.get(field, {}).items():
            if models.get(name, {}).get(field) != expected:
                raise ValueError(f'Policy binding does not match runtime {field} for {name}')


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
    _validate_metadata(value, names)
    if not allow_exploratory:
        if value.get('deployment_ready') is not True:
            raise ValueError('Exploratory policy requires --allow-exploratory')
    return copy.deepcopy(value)
