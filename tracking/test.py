import argparse
import os
import random
import sys
from os.path import dirname, exists, join, realpath

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
from tqdm import tqdm

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from lib.core.config import config as cfg
from lib.core.eval_otb import eval_auc_tune
from lib.eval_toolkit.pysot.datasets import VOTDataset
from lib.eval_toolkit.pysot.evaluation import EAOBenchmark
from lib.models.model import SiamBC
from lib.tracker.siambc import SiamBCTracker
from lib.utils.utils import cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou
from tracking.set_config_by_type import set_config_by_type


def parse_args():
    parser = argparse.ArgumentParser(description='Test SiamBC')
    parser.add_argument('--resume', default="snapshot/siambc.pth", type=str, help='pretrained model')
    parser.add_argument('--dataset', default='VOT2019', help='dataset test')
    parser.add_argument('--epoch_test', default=False, type=bool, help='multi-gpu epoch test flag')
    parser.add_argument('--align', default=True, type=bool, help='alignment module flag')  # bool
    parser.add_argument('--type', default=None, type=str, help='type flag')
    parser.add_argument('--video', default=None, type=str, help='test a video in benchmark')
    parser.add_argument('--debug', default=False, type=bool, help='debug模式')
    parser.add_argument('--gpu_id', default='-1', type=str)

    args = parser.parse_args()

    if args.gpu_id != '-1':
        print("==> setting gpu ", args.gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    return args


class TestTracker:
    def __init__(self, siam_tracker, siam_net, dataset, args):
        self.siam_tracker, self.siam_net, self.args = siam_tracker, siam_net, args
        self.dataset = dataset

        self.total_f = 0
        self.total_toc = 0
        self.name = self.args.resume.split('/')[-1].split('.')[0]

    def track(self, v, index=1):
        video = self.dataset[v]
        start_frame, toc = 0, 0

        # save result to evaluate
        if self.args.epoch_test:
            suffix = self.args.resume.split('/')[-1]
            suffix = suffix.split('.')[0]
            tracker_path = os.path.join('result', self.args.dataset, suffix)
        else:
            tracker_path = os.path.join('result', self.args.dataset)

        if not os.path.exists(tracker_path):
            os.makedirs(tracker_path)

        if 'VOT' in self.args.dataset:
            baseline_path = os.path.join(tracker_path, 'baseline')
            video_path = os.path.join(baseline_path, video['name'])
            if not os.path.exists(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, video['name'] + '_001.txt')
        else:
            result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['name']))

        if os.path.exists(result_path):
            return  # for mult-gputesting

        regions = []
        lost = 0

        image_files, gt = video['image_files'], video['gt']

        for f, image_file in enumerate(image_files):
            im = cv2.imread(image_file)
            # rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if len(im.shape) == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # align with training

            tic = cv2.getTickCount()
            if f == start_frame:  # init
                cx, cy, w, h = get_axis_aligned_bbox(gt[f])

                target_pos = np.array([cx, cy])
                target_sz = np.array([w, h])

                state = self.siam_tracker.init(im, target_pos, target_sz, self.siam_net)  # init tracker

                regions.append(1 if 'VOT' in self.args.dataset else gt[f])
            elif f > start_frame:  # tracking
                state = self.siam_tracker.track(state, im)

                location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                b_overlap = poly_iou(gt[f], location) if 'VOT' in self.args.dataset else 1

                if b_overlap > 0:
                    regions.append(location)
                else:
                    regions.append(2)
                    start_frame = f + 5
                    lost += 1
            else:
                regions.append(0)

            toc += cv2.getTickCount() - tic

        with open(result_path, "w") as fin:
            if 'VOT' in self.args.dataset:
                for x in regions:
                    if isinstance(x, int):
                        fin.write("{:d}\n".format(x))
                    else:
                        p_bbox = x.copy()
                        fin.write(','.join([str(i) for i in p_bbox]) + '\n')
            elif 'OTB' in self.args.dataset or 'LASOT' in self.args.dataset:
                for x in regions:
                    p_bbox = x.copy()
                    fin.write(
                        ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')
            elif 'VISDRONE' in self.args.dataset or 'GOT10K' in self.args.dataset:
                for x in regions:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for idx, i in enumerate(p_bbox)]) + '\n')
            elif 'UAV' in self.args.dataset or 'NFS' in self.args.dataset:
                for x in regions:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for idx, i in enumerate(p_bbox)]) + '\n')

        toc /= cv2.getTickFrequency()
        print('({:3d}) [{:25s}] Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps  Lost {}'
              .format(index, self.args.resume, video['name'], toc, f / toc, lost))

        self.total_f += f
        self.total_toc += toc

        return lost

    def info(self):
        if self.total_toc == 0:
            return
        print("[{}] Average Speed: {:3.1f}fps".format(self.name, self.total_f / self.total_toc))


def main():
    args = parse_args()

    info = edict()
    info.dataset = args.dataset
    info.epoch_test = args.epoch_test

    siam_info = edict()
    siam_info.dataset = args.dataset
    siam_info.epoch_test = args.epoch_test

    siam_info.align = True if 'VOT' in args.dataset and args.align == 'True' else False

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # configurate parameters by settings type
    set_config_by_type(args.type)

    siam_tracker = SiamBCTracker(siam_info)
    siam_net = SiamBC(align=siam_info.align)
    # print(siam_net)
    print('===> init Siamese <====')

    siam_net.eval()
    siam_net = siam_net.cuda()

    print('====> warm up <====')
    for _ in tqdm(range(100)):
        siam_net.template(torch.rand(1, 3, 127, 127).cuda())
        siam_net.track(torch.rand(1, 3, 255, 255).cuda())

    # prepare video
    dataset = load_dataset(args.dataset)
    print(f"load {args.dataset}: len {len(dataset)}")
    video_keys = list(dataset.keys()).copy()

    test_tracker = TestTracker(siam_tracker, siam_net, dataset, args)
    if args.video is not None:
        print(f"args.video: {args.video}")
        test_tracker.track(args.video)
    else:
        total_lost = 0
        index = 1
        for video in video_keys:
            test_tracker.track(video, index)
            index += 1
        print("[{:27s}] total lost: {:d}".format(args.resume, total_lost))
        test_tracker.info()


def track_tune(tracker, net, video, config):
    benchmark_name = config['benchmark']
    resume = config['resume']
    hp = config['hp']  # scale_step, scale_penalty, scale_lr, window_influence

    if cfg.DEBUG:
        tracker_path = "debug"
    else:
        tracker_path = join('test', (benchmark_name + resume.split('/')[-1].split('.')[0] +
                                     '_small_size_{:.4f}'.format(hp['small_sz']) +
                                     '_big_size_{:.4f}'.format(hp['big_sz']) +
                                     '_ratio_{:.4f}'.format(hp['ratio']) +
                                     '_penalty_k_{:.4f}'.format(hp['penalty_k']) +
                                     '_w_influence_{:.4f}'.format(hp['window_influence']) +
                                     '_scale_lr_{:.4f}'.format(hp['lr'])).replace('.', '_'))  # no .
    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in benchmark_name:
        baseline_path = join(tracker_path, 'baseline')
        video_path = join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = join(video_path, video['name'] + '_001.txt')
    elif 'GOT10K' in benchmark_name:
        re_video_path = os.path.join(tracker_path, video['name'])
        if not exists(re_video_path): os.makedirs(re_video_path)
        result_path = os.path.join(re_video_path, '{:s}.txt'.format(video['name']))
    else:
        result_path = join(tracker_path, '{:s}.txt'.format(video['name']))

    # occ for parallel running
    if not os.path.exists(result_path):
        fin = open(result_path, 'w')
        fin.close()
    else:
        if benchmark_name.startswith('OTB') or 'LASOT' in benchmark_name:
            return tracker_path
        elif benchmark_name.startswith('VOT') or benchmark_name.startswith('GOT10K'):
            return 0
        elif 'UAV' in benchmark_name:
            return tracker_path
        else:
            raise NotImplementedError('benchmark not supported now')

    start_frame, lost_times, toc = 0, 0, 0

    regions = []  # result and states[1 init / 2 lost / 0 skip]
    image_files, gt = video['image_files'], video['gt']

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            state = tracker.init(im, target_pos, target_sz, net, hp=hp)  # init tracker
            regions.append([float(1)] if 'VOT' in benchmark_name else gt[f])
        elif f > start_frame:  # tracking
            state = tracker.track(state, im)  # track
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            b_overlap = poly_iou(gt[f], location) if 'VOT' in benchmark_name else 1
            if b_overlap > 0:
                regions.append(location)
            else:
                regions.append([float(2)])
                lost_times += 1
                start_frame = f + 5  # skip 5 frames
        else:  # skip
            regions.append([float(0)])

    # save results for OTB
    if 'OTB' in benchmark_name or 'LASOT' in benchmark_name or 'UAV' in benchmark_name:
        with open(result_path, "w") as fin:
            for x in regions:
                p_bbox = x.copy()
                fin.write(
                    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')
    elif 'VISDRONE' in benchmark_name or 'GOT10K' in benchmark_name:
        with open(result_path, "w") as fin:
            for x in regions:
                p_bbox = x.copy()
                fin.write(','.join([str(i) for idx, i in enumerate(p_bbox)]) + '\n')
    elif 'VOT' in benchmark_name:
        with open(result_path, "w") as fin:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')

    if 'OTB' in benchmark_name or 'LASOT' in benchmark_name or 'VIS' in benchmark_name or 'VOT' in benchmark_name or 'GOT10K' in benchmark_name \
        or 'UAV' in benchmark_name:
        return tracker_path
    else:
        raise NotImplementedError('benchmark not supported now')


def auc_otb(tracker, net, config):
    """
    get AUC for OTB benchmark
    """
    dataset = load_dataset(config['benchmark'])
    video_keys = list(dataset.keys()).copy()
    random.shuffle(video_keys)

    for video in video_keys:
        result_path = track_tune(tracker, net, dataset[video], config)

    auc = eval_auc_tune(result_path, config['benchmark'])

    return auc


def auc_uav(tracker, net, config):
    """
    get AUC for UAV123 benchmark
    """
    dataset = load_dataset(config['benchmark'])
    video_keys = list(dataset.keys()).copy()
    random.shuffle(video_keys)

    for video in video_keys:
        result_path = track_tune(tracker, net, dataset[video], config)

    auc = eval_auc_tune(result_path, config['benchmark'])

    return auc


def auc_lasot(tracker, net, config):
    """
    get AUC for LASOT benchmark
    """
    dataset = load_dataset(config['benchmark'])
    video_keys = list(dataset.keys()).copy()
    random.shuffle(video_keys)

    for video in video_keys:
        result_path = track_tune(tracker, net, dataset[video], config)

    auc = eval_auc_tune(result_path, config['benchmark'])

    return auc


def eao_vot(tracker, net, config):
    dataset = load_dataset(config['benchmark'])
    video_keys = sorted(list(dataset.keys()).copy())

    for video in video_keys:
        result_path = track_tune(tracker, net, dataset[video], config)

    re_path = result_path.split('/')[0]
    tracker = result_path.split('/')[-1]

    # give abs path to json path
    data_path = join(realpath(dirname(__file__)), '../dataset')
    dataset = VOTDataset(config['benchmark'], data_path)

    dataset.set_tracker(re_path, tracker)
    benchmark = EAOBenchmark(dataset)
    eao = benchmark.eval(tracker)
    eao = eao[tracker]['all']

    return eao


if __name__ == '__main__':
    main()
