"""Check SBI model and one training sample before starting a full run."""

import argparse
import os

import torch
import yaml

from dataset.sbi_dataset import SBIDataset
from detectors import DETECTOR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector-path', default='training/config/detector/sbi.yaml')
    parser.add_argument('--weights', help='EfficientNet-B4 pretrained weights file')
    parser.add_argument('--data-mode', choices=['rgb', 'lmdb'], default='rgb')
    args = parser.parse_args()

    with open(args.detector_path, encoding='utf-8') as file:
        config = yaml.safe_load(file)
    with open('training/config/train_config.yaml', encoding='utf-8') as file:
        config.update(yaml.safe_load(file))
    if args.weights:
        config['pretrained'] = args.weights
    config['lmdb'] = args.data_mode == 'lmdb'
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'

    assert config['model_name'] == 'sbi' and config['dataset_type'] == 'blend'
    assert config['backbone_config']['num_classes'] == 2
    model = DETECTOR['sbi'](config).eval()
    with torch.no_grad():
        output = model({'image': torch.zeros(1, 3, config['resolution'], config['resolution'])}, inference=True)
    assert output['cls'].shape == (1, 2), output['cls'].shape
    print('Model check passed: binary output shape (1, 2)')

    dataset = SBIDataset(config, mode='train')
    assert len(dataset) > 0, 'No real training frames found'
    image_path, label = dataset.real_imglist[0]
    assert label == 0
    landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')
    if config['lmdb']:
        if landmark_path.startswith('.'):
            landmark_path = landmark_path.replace('./datasets\\', '')
        with dataset.env.begin(write=False) as txn:
            assert txn.get(landmark_path.encode()) is not None, f'Landmark LMDB key missing: {landmark_path}'
    else:
        full_path = landmark_path if landmark_path.startswith('.') else os.path.join(config['rgb_dir'], landmark_path)
        assert os.path.isfile(full_path), f'Landmark file missing: {full_path}'
    batch = dataset.collate_fn([dataset[0]])
    assert batch['image'].shape == (2, 3, config['resolution'], config['resolution'])
    assert batch['label'][0].item() == 0 and set(batch['label'].tolist()) <= {0, 1}
    print('Dataset check passed: real/SBI pair and landmark available')


if __name__ == '__main__':
    main()
