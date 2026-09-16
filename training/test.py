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
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        feature_lists += list(predictions['feat'].cpu().detach().numpy())
    
    return np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)

# 控制「要測哪些 dataset」
def test_epoch(model, test_data_loaders, output_dir):
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
        predictions_nps, label_nps,feat_nps = test_one_dataset(model, test_data_loaders[key])
        
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset

        prediction_path = output_path / f'{key}_predictions.csv'
        with prediction_path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['image_path', 'label', 'probability'])
            writer.writerows(zip(data_dict['image'], label_nps.tolist(),
                                 np.asarray(predictions_nps).reshape(-1).tolist()))
        metric_path = output_path / f'{key}_metrics.json'
        scalar_metrics = {name: float(value) for name, value in metric_one_dataset.items()
                          if name not in ('pred', 'label')}
        with metric_path.open('w', encoding='utf-8') as file:
            json.dump(scalar_metrics, file, indent=2, sort_keys=True)
        
        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in scalar_metrics.items():
            tqdm.write(f"{k}: {v}")
        tqdm.write(f"Predictions saved to {prediction_path}")
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
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # start testing
    best_metric = test_epoch(model, test_data_loaders, args.output_dir)
    print('===> Test Done!')

if __name__ == '__main__':
    main()
