# -*- encoding: utf-8 -*-

import argparse
import os
import time

from mpi4py import MPI

parser = argparse.ArgumentParser(description='multi-gpu test all epochs')
parser.add_argument('--name', default='siambc', type=str, help='tracker名称')
parser.add_argument('--base', default='/data/SiamBC/', type=str, help='基础路径')
parser.add_argument('--start_epoch', '-s', default=10, type=int, help='test start epoch')
parser.add_argument('--end_epoch', '-e', default=20, type=int, help='test end epoch')
parser.add_argument('--threads', '-t', default=4, type=int)
parser.add_argument('--gpu_nums', '-gn', default=4, type=int, help='gpu numbers')
parser.add_argument('--type', default='nonlocal_cascade', type=str, help='tracker类型')
parser.add_argument('--dataset', '-d', default='GOT10k', type=str, help='benchmark to test, dataset name')
parser.add_argument('-add', default=0, type=int, help='应对GPU被占用的情况')
args = parser.parse_args()

cur_path = os.path.abspath(os.path.dirname(__file__))

dataset = {
    'GOT10k'     : 'got',
    'OTB100'     : 'otb',
    'TrackingNet': 'trackingnet',
}

"""
运行命令
[got10k]
nohup mpiexec --allow-run-as-root -n 8 \
    python -u got10k_toolkit/toolkit/test_epochs.py \
    --base /data/trackit_nonlocal_light_fix \
    --type deform_light_nonlocal_fix \
    -s 30 -e 39 -gn 4 -t 8 \
    >test_got.log 2>&1 &

[otb100]
nohup mpiexec --allow-run-as-root -n 1 python -u got10k_toolkit/toolkit/test_epochs.py -s 49 -e 49 -gn 1 -t 1 -d OTB100 >test_otb.log 2>&1 &
"""

# init gpu and epochs
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
GPU_ID = rank % args.gpu_nums + args.add
node_name = MPI.Get_processor_name()  # get the name of the node
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

print(f"node name: {node_name}, GPU_ID: {GPU_ID}")
time.sleep(rank * 5)

# run test scripts -- one epoch for each thread
for i in range((args.end_epoch - args.start_epoch + 1) // args.threads + 1):
    try:
        epoch_ID += args.threads
    except:
        epoch_ID = rank % (args.end_epoch - args.start_epoch + 1) + args.start_epoch

    if epoch_ID > args.end_epoch:
        continue

    snapshot = f'snapshot/checkpoint_e{epoch_ID}.pth'
    print(f'==> test epoch {epoch_ID}')
    os.system(
        f'python -u {cur_path}/test_{dataset[args.dataset]}.py --name {args.name} --base {args.base} '
        f'--snapshot {snapshot} --type {args.type} --gpu_id {GPU_ID}'
    )
