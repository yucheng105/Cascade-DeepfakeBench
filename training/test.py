"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
import csv
import json
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger

PREDICTION_CSV_COLUMNS = [
    'sample_id',
    'dataset',
    'label',
    'detector_name',
    'modality',
    'fake_score',
    'score_type',
    'inference_time_ms',
    'window_start_sec',
    'window_end_sec',
    'status',
    'error_message',
]

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, 
                    default='/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth')
parser.add_argument('--output_dir', type=str, default='./logs/testing',
                    help='directory for prediction CSV and metric JSON files')
#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    np.random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    # 給我一個 dataset 名稱，我幫你建立那個 dataset 的 DataLoader。
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )
        # 把 test_set 包裝成 DataLoader，這樣就可以用 batch 的方式讀取資料
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring

# 控制「這個 dataset 的所有 batch 怎麼測」
def test_one_dataset(model, data_loader):
    prediction_lists = []   # fake score
    feature_lists = []      # feature vector
    label_lists = []        # ground truth labels
    inference_time_lists = []  # model-forward time allocated to each item
    # 一個 dataloader 是一個 batch
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']

        # convert label to binary
        label = torch.where(data_dict['label'] != 0, 1, 0)

        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        # inference 結果移併回傳到 predictions dict
        # Synchronization is needed for meaningful CUDA timings. Data loading and
        # host-to-device copies are intentionally excluded from inference time.
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        predictions = inference(model, data_dict)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        batch_inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        batch_size = int(data_dict['label'].shape[0])
        inference_time_lists.extend([batch_inference_time_ms / batch_size] * batch_size)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        # feature_lists += list(predictions['feat'].cpu().detach().numpy())     # 先不保留特徵，個人電腦記憶體不足
    
    return (np.array(prediction_lists), np.array(label_lists),
            np.array(feature_lists), np.array(inference_time_lists))

# 控制「要測哪些 dataset」
def sample_id_from_image_path(image_path):
    """Return the source video filename stem represented by a frame path."""
    if isinstance(image_path, (list, tuple)):
        if not image_path:
            raise ValueError('Cannot derive sample_id from an empty frame list.')
        image_path = image_path[0]

    normalized_path = str(image_path).replace('\\', '/').rstrip('/')
    path_parts = normalized_path.split('/')
    if len(path_parts) < 2:
        raise ValueError('Cannot derive sample_id from path: {}'.format(image_path))
    return path_parts[-2]


def aggregate_video_predictions(data_dict, predictions, labels, inference_times_ms,
                                dataset_name, detector_name):
    """Aggregate frame/clip predictions into one CSV row per source video."""
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    inference_times_ms = np.asarray(inference_times_ms).reshape(-1)
    image_paths = data_dict['image']
    video_names = data_dict['video_name']

    lengths = {
        len(predictions), len(labels), len(inference_times_ms),
        len(image_paths), len(video_names),
    }
    if len(lengths) != 1:
        raise ValueError(
            'Predictions, labels, timings, image paths, and video names must be aligned.'
        )

    videos = {}
    for video_name, image_path, prediction, label, elapsed_ms in zip(
            video_names, image_paths, predictions, labels, inference_times_ms):
        binary_label = int(label != 0)
        sample_id = sample_id_from_image_path(image_path)
        if video_name not in videos:
            videos[video_name] = {
                'sample_id': sample_id,
                'label': binary_label,
                'scores': [],
                'inference_times_ms': [],
            }
        elif videos[video_name]['label'] != binary_label:
            raise ValueError('Inconsistent labels for video: {}'.format(video_name))
        elif videos[video_name]['sample_id'] != sample_id:
            raise ValueError('Inconsistent sample IDs for video: {}'.format(video_name))

        videos[video_name]['scores'].append(float(prediction))
        videos[video_name]['inference_times_ms'].append(float(elapsed_ms))

    rows = []
    for video_name, video in videos.items():
        rows.append({
            'sample_id': video['sample_id'],
            'dataset': dataset_name,
            'label': video['label'],
            'detector_name': detector_name,
            'modality': 'visual',
            'fake_score': float(np.mean(video['scores'])),
            'score_type': 'probability',
            'inference_time_ms': float(np.sum(video['inference_times_ms'])),
            'window_start_sec': '',
            'window_end_sec': '',
            'status': 'ok',
            'error_message': '',
            '_video_name': video_name,
        })

    # The dataset class shuffles test items. Sort here so repeated runs produce
    # stable CSVs: real first, then fake; FF++ manipulation groups stay distinct.
    rows.sort(key=lambda row: (row['label'], row['_video_name']))
    for row in rows:
        del row['_video_name']
    return rows


def write_prediction_csv(prediction_path, rows):
    with prediction_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=PREDICTION_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# Test each configured dataset and write one video-level CSV per dataset.
def test_epoch(model, test_data_loaders, output_dir, detector_name):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # get names for all testing datasets
    keys = test_data_loaders.keys()
    for key in keys:
        # get the data dictionary for the current dataset
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, label_nps, feat_nps, inference_time_nps = \
            test_one_dataset(model, test_data_loaders[key])
        
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset

        prediction_path = output_path / f'{key}_predictions.csv'
        prediction_rows = aggregate_video_predictions(
            data_dict=data_dict,
            predictions=predictions_nps,
            labels=label_nps,
            inference_times_ms=inference_time_nps,
            dataset_name=key,
            detector_name=detector_name,
        )
        write_prediction_csv(prediction_path, prediction_rows)
        metric_path = output_path / f'{key}_metrics.json'
        scalar_metrics = {name: float(value) for name, value in metric_one_dataset.items()
                          if name not in ('pred', 'label')}
        with metric_path.open('w', encoding='utf-8') as file:
            json.dump(scalar_metrics, file, indent=2, sort_keys=True)
        
        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in scalar_metrics.items():
            tqdm.write(f"{k}: {v}")
        tqdm.write(f"Predictions saved to {prediction_path} ({len(prediction_rows)} videos)")
        tqdm.write(f"Metrics saved to {metric_path}")

    return metrics_all_datasets

# 把一個 batch 的 data_dict 丟進 detector，拿回 predictions。
@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    cudnn.benchmark = False
    cudnn.deterministic = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        # load the pre-trained weights
        # ckpt = torch.load(weights_path, map_location=device, weights_only=True)
        ckpt = torch.load(weights_path, map_location=device)

        # for sbi .tar weights
        if config['model_name'] == 'sbi':
            # Map the downloaded SBI checkpoint to this detector's parameter names.
            state_dict = ckpt["model"]
            mapped = {}
            for key, value in state_dict.items():
                if key.startswith("net._fc."):
                    key = key.replace("net._fc.", "backbone.last_layer.", 1)
                elif key.startswith("net."):
                    key = key.replace("net.", "backbone.efficientnet.", 1)
                mapped[key] = value
            model.load_state_dict(mapped, strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # start testing
    best_metric = test_epoch(
        model, test_data_loaders, args.output_dir, config['model_name']
    )
    print('===> Test Done!')

if __name__ == '__main__':
    main()
