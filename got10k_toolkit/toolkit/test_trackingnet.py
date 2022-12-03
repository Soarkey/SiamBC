import argparse
import os

from got10k_toolkit.toolkit.experiments import ExperimentTrackingNet
from got10k_toolkit.toolkit.trackers.identity_tracker import IdentityTracker
from got10k_toolkit.toolkit.trackers.net_wrappers import NetWithBackbone

parser = argparse.ArgumentParser(description='test trackingnet')
parser.add_argument('--name', '-n', default='siamx', type=str, help='tracker名称')
parser.add_argument('--base', '-b', default='/data/siambc/', type=str, help='基础路径')
parser.add_argument('--snapshot', '-s', default='snapshot/checkpoint_e46.pth', type=str, help='snapshot path')
parser.add_argument('--type', '-t', default='base', type=str, help='type')
parser.add_argument('--gpu_id', default=-1, type=int, help='gpu id')
parser.add_argument('--subset', default='test', type=str, help='train|val|test')
args = parser.parse_args()

# Specify the path
net_path = os.path.join(args.base, args.snapshot)
dataset_root = '/data/datasets/testing_dataset/TrackingNet/'  # Absolute path of the datasets

if args.gpu_id != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Tracker
# 参数
kwargs = {
    'arch'   : 'Ocean',
    'dataset': 'TrackingNet',
    'align'  : True,
    'type'   : args.type
}

net = NetWithBackbone(net_path=net_path, use_gpu=True, **kwargs)
tracker = IdentityTracker(name='siamx', net=net, **kwargs)

epoch = args.snapshot.split('/')[-1].split('.')[0]

# Test
experiment = ExperimentTrackingNet(
    root_dir=dataset_root,  # TrackingNet's root directory
    subset=args.subset,  # 'train' | 'val' | 'test'
    result_dir=f'got_toolkit_results/{args.name}/TrackingNet/{epoch}',  # where to store tracking results
    report_dir=f'got_toolkit_reports/{args.name}/TrackingNet'  # where to store evaluation reports
)
experiment.run(tracker, visualize=False)
