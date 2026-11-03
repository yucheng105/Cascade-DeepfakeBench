"""Run the Xception-to-UCF sequential cascade.

Docker example:

python cascade/test.py \
    --low_threshold 0.2 \
    --high_threshold 0.8 \
    --final_threshold 0.5
"""

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / 'training'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from cascade.data import VideoIdentityDataset
from cascade.config import validate_shared_input_contract
from cascade.inference import run_frame_detector_stage
from cascade.metrics import final_classification_metrics, operational_auc
from cascade.video_scoring import (
    UNCERTAIN,
    merge_cascade_decisions,
    route_final_stage,
    route_video_scores,
    validate_double_threshold,
    validate_final_threshold,
)
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR


def project_path(value):
    """Resolve command-line paths consistently inside the Docker mount."""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the Xception-to-UCF sequential cascade.'
    )
    parser.add_argument(
        '--detector_path', '--stage1_detector_path',
        dest='stage1_detector_path',
        default='training/config/detector/xception.yaml',
    )
    parser.add_argument(
        '--test_config',
        default='training/config/test_config.yaml',
    )
    parser.add_argument(
        '--weights_path', '--stage1_weights_path',
        dest='stage1_weights_path',
        default='training/weights/xception_best.pth',
    )
    parser.add_argument(
        '--stage2_detector_path',
        default='training/config/detector/ucf.yaml',
    )
    parser.add_argument(
        '--stage2_weights_path',
        default='training/weights/ucf_best.pth',
    )
    parser.add_argument('--test_dataset', default='FaceForensics++')
    parser.add_argument('--low_threshold', type=float, required=True)
    parser.add_argument('--high_threshold', type=float, required=True)
    parser.add_argument('--final_threshold', type=float, required=True)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--stage2_batch_size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--device')
    return parser.parse_args()


def load_config(
    detector_path,
    test_config_path,
    test_dataset,
    batch_size=None,
    workers=None,
):
    with project_path(detector_path).open('r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    with project_path(test_config_path).open('r', encoding='utf-8') as file:
        config.update(yaml.safe_load(file))

    config['test_dataset'] = test_dataset
    if batch_size is not None:
        config['test_batchSize'] = batch_size
    if workers is not None:
        config['workers'] = workers

    config['pretrained'] = str(project_path(config['pretrained']))
    config['dataset_json_folder'] = str(
        project_path(config['dataset_json_folder'])
    )
    config['rgb_dir'] = str(project_path(config['rgb_dir']))
    config['lmdb_dir'] = str(project_path(config['lmdb_dir']))
    return config


def initialize_seed(config):
    seed = config.get('manualSeed')
    if seed is None:
        seed = 1024
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_argument):
    if device_argument:
        device = torch.device(device_argument)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available in Docker.')
    return device


def build_data_loader(config, device):
    base_dataset = DeepfakeAbstractBaseDataset(config=config, mode='test')
    dataset = VideoIdentityDataset(base_dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config['test_batchSize'],
        shuffle=False,
        num_workers=int(config['workers']),
        collate_fn=dataset.collate_fn,
        drop_last=False,
        pin_memory=device.type == 'cuda',
    )
    return dataset, data_loader


def build_subset_data_loader(dataset, video_names, config, device):
    subset = dataset.subset_for_videos(video_names)
    return torch.utils.data.DataLoader(
        dataset=subset,
        batch_size=config['test_batchSize'],
        shuffle=False,
        num_workers=int(config['workers']),
        collate_fn=dataset.collate_fn,
        drop_last=False,
        pin_memory=device.type == 'cuda',
    )


def build_model(config, weights_path, device):
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    checkpoint = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    return model


def format_optional_metric(value):
    return 'N/A' if value is None else '{:.6f}'.format(value)


def print_stage_summary(
    stage_number,
    config,
    video_scores,
    routed_videos,
    threshold_description,
):
    counts = Counter(video.decision for video in routed_videos)
    labels = Counter(video.label for video in video_scores)
    auc = operational_auc(video_scores)
    frame_count = sum(video.frame_count for video in video_scores)

    print(
        '\n=== Cascade Stage {}: {} ==='.format(
            stage_number, config['model_name']
        )
    )
    print('dataset: {}'.format(config['test_dataset']))
    print('frames: {}'.format(frame_count))
    print('videos: {}'.format(len(video_scores)))
    print('ground_truth_REAL: {}'.format(labels[0]))
    print('ground_truth_FAKE: {}'.format(labels[1]))
    print(threshold_description)
    print('REAL: {}'.format(counts['REAL']))
    if stage_number == 1:
        print('UNCERTAIN: {}'.format(counts['UNCERTAIN']))
    print('FAKE: {}'.format(counts['FAKE']))
    print('AUC: {}'.format(format_optional_metric(auc)))


def print_final_summary(final_metrics):
    print('\n=== Final Cascade Metrics ===')
    print('videos: {}'.format(final_metrics['video_count']))
    print('ground_truth_REAL: {}'.format(final_metrics['real_count']))
    print('ground_truth_FAKE: {}'.format(final_metrics['fake_count']))
    print('ACC: {}'.format(format_optional_metric(final_metrics['acc'])))
    print('FNR (FAR, fake accepted as real): {}'.format(format_optional_metric(final_metrics['fnr'])))
    print('FPR (FRR, real rejected as fake): {}'.format(format_optional_metric(final_metrics['fpr'])))


def main():
    args = parse_args()
    validate_double_threshold(args.low_threshold, args.high_threshold)
    validate_final_threshold(args.final_threshold)
    os.chdir(str(PROJECT_ROOT))
    # xception.yaml + test_config.yaml → stage1_config
    stage1_config = load_config(
        detector_path=args.stage1_detector_path,
        test_config_path=args.test_config,
        test_dataset=args.test_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    # ucf.yaml + test_config.yaml → stage2_config
    stage2_config = load_config(
        detector_path=args.stage2_detector_path,
        test_config_path=args.test_config,
        test_dataset=args.test_dataset,
        batch_size=args.stage2_batch_size,
        workers=args.workers,
    )

    # 先寫死
    if stage1_config['model_name'] != 'xception':
        raise ValueError('Stage 1 detector must be xception.')
    if stage2_config['model_name'] != 'ucf':
        raise ValueError('Stage 2 detector must be ucf.')
    
    # check if detectors can use the same dataset output
    validate_shared_input_contract(stage1_config, stage2_config)
    initialize_seed(stage1_config)
    device = choose_device(args.device)

    dataset, stage1_data_loader = build_data_loader(stage1_config, device)
    stage1_model = build_model(
        stage1_config,
        project_path(args.stage1_weights_path),
        device,
    )
    stage1_scores = run_frame_detector_stage(
        model=stage1_model,
        data_loader=stage1_data_loader,
        device=device,
        description='Stage 1: Xception',
    )
    stage1_routed = route_video_scores(
        stage1_scores,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )
    print_stage_summary(
        stage_number=1,
        config=stage1_config,
        video_scores=stage1_scores,
        routed_videos=stage1_routed,
        threshold_description='thresholds: low={}, high={}'.format(
            args.low_threshold, args.high_threshold
        ),
    )

    uncertain_video_names = [
        video.video_name
        for video in stage1_routed
        if video.decision == UNCERTAIN
    ]
    del stage1_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # only move on to the next stage if there are uncertain videos
    if uncertain_video_names:
        stage2_data_loader = build_subset_data_loader(
            dataset=dataset,
            video_names=uncertain_video_names,
            config=stage2_config,
            device=device,
        )
        stage2_model = build_model(
            stage2_config,
            project_path(args.stage2_weights_path),
            device,
        )
        stage2_scores = run_frame_detector_stage(
            model=stage2_model,
            data_loader=stage2_data_loader,
            device=device,
            description='Stage 2: UCF',
        )
        stage2_routed = route_final_stage(
            stage2_scores, args.final_threshold
        )
        del stage2_model
    else:
        stage2_scores = ()
        stage2_routed = ()

    print_stage_summary(
        stage_number=2,
        config=stage2_config,
        video_scores=stage2_scores,
        routed_videos=stage2_routed,
        threshold_description='threshold: {}'.format(args.final_threshold),
    )

    final_videos = merge_cascade_decisions(stage1_routed, stage2_routed)
    print_final_summary(final_classification_metrics(final_videos))

if __name__ == '__main__':
    main()
