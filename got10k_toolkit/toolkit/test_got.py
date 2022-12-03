# -*- encoding: utf-8 -*-
import argparse
import os

from got10k_toolkit.toolkit.experiments import ExperimentGOT10k
from got10k_toolkit.toolkit.trackers.identity_tracker import IdentityTracker
from got10k_toolkit.toolkit.trackers.net_wrappers import NetWithBackbone

parser = argparse.ArgumentParser(description='test got10k')
parser.add_argument('--name', '-n', default='siamx', type=str, help='tracker')
parser.add_argument('--base', '-b', default='/data/SiamBC/', type=str, help='file base path')
parser.add_argument('--snapshot', '-s', default='snapshot/checkpoint_e50.pth', type=str, help='snapshot path')
parser.add_argument('--type', '-t', default='nonlocal_cascade', type=str, help='type')
parser.add_argument('--gpu_id', default=-1, type=int, help='gpu id')
parser.add_argument('--subset', default='test', type=str, help='train|val|test')
args = parser.parse_args()

# Run
"""
nohup python -u got10k_toolkit/toolkit/test_got.py \
 --base /data/SiamBC/ \
 --snapshot snapshot/checkpoint_e28.pth \
 --type base \
 --gpu_id 1 \
 --subset val \
 > test_got_val.log 2>&1 &
"""

# Specify the path
net_path = os.path.join(args.base, args.snapshot)
dataset_root = '/data/datasets/testing_dataset/GOT10K'  # Absolute path of the datasets

# For val
if args.subset == 'val':
    dataset_root = '/data/dataset/got_10k/val/'

if args.gpu_id != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Tracker Params
kwargs = {
    'dataset': 'GOT10K',
    'align'  : True,
    'type'   : args.type
}

tracker_name = args.snapshot.split('/')[-1].split('.')[0]  # checkpoint_e46
net = NetWithBackbone(net_path=net_path, use_gpu=True, **kwargs)
tracker = IdentityTracker(name=tracker_name, net=net, **kwargs)

# Test
experiment = ExperimentGOT10k(
    root_dir=dataset_root,  # GOT-10k's root directory
    subset=args.subset,  # 'train' | 'val' | 'test'
    result_dir=f'got_toolkit_results/{args.name}',  # where to store tracking results
    report_dir=f'got_toolkit_reports/{args.name}'  # where to store evaluation reports
)
experiment.run(tracker, visualize=False)
experiment.report(tracker_names=[tracker_name], plot_curves=False)
