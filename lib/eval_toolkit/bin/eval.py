import argparse
from multiprocessing import Pool

from tqdm import tqdm

from lib.eval_toolkit.pysot.datasets import LaSOTDataset, NFSDataset, OTBDataset, UAVDataset, VOTDataset, VOTLTDataset
from lib.eval_toolkit.pysot.evaluation import AccuracyRobustnessBenchmark, EAOBenchmark, F1Benchmark, OPEBenchmark

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Object Tracking Evaluation')
    parser.add_argument('--dataset_dir', type=str, help='dataset root directory')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--tracker_result_dir', type=str, help='tracker result root')
    parser.add_argument('--trackers', nargs='+')
    parser.add_argument('--vis', dest='vis', default=False)
    parser.add_argument('--show_video_level', dest='show_video_level', default=False)
    parser.add_argument('--num', type=int, help='number of processes to eval', default=24)
    args = parser.parse_args()

    tracker_dir = args.tracker_result_dir
    trackers = args.trackers
    root = args.dataset_dir

    if args.vis:
        from lib.eval_toolkit.pysot.visualization import draw_success_precision, draw_f1

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))
    trackers = ['checkpoint_e' + x for x in trackers]

    if 'OTB' in args.dataset:
        dataset = OTBDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)
    elif 'LaSOT' == args.dataset or "LASOT" == args.dataset:
        dataset = LaSOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        # success_ret = benchmark.eval_success(trackers)
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        norm_precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                                                trackers), desc='eval norm precision', total=len(trackers), ncols=100):
                norm_precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                              show_video_level=args.show_video_level)
        if args.vis:
            draw_success_precision(success_ret,
                                   name=dataset.name,
                                   videos=dataset.attr['ALL'],
                                   attr='ALL',
                                   precision_ret=precision_ret,
                                   norm_precision_ret=norm_precision_ret)
    elif 'UAV' in args.dataset:
        dataset = UAVDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)
    elif 'NFS' in args.dataset:
        dataset = NFSDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)
    elif 'VOT' in args.dataset:
        dataset = VOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        ar_benchmark = AccuracyRobustnessBenchmark(dataset)
        ar_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(ar_benchmark.eval,
                                                trackers), desc='eval ar', total=len(trackers), ncols=100):
                ar_result.update(ret)
        # benchmark.show_result(ar_result)
        benchmark = EAOBenchmark(dataset, tags=['all', 'occlusion', 'motion_change', 'size_change',
                                                'illum_change', 'camera_motion', 'empty'])
        # benchmark = EAOBenchmark(dataset)
        eao_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                                                trackers), desc='eval eao', total=len(trackers), ncols=100):
                eao_result.update(ret)
        # benchmark.show_result(eao_result)

        import json

        print(json.dumps(eao_result, indent=2))

        ar_benchmark.show_result(ar_result, eao_result,
                                 show_video_level=args.show_video_level)
    elif 'VOT2018-LT' == args.dataset:
        dataset = VOTLTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = F1Benchmark(dataset)
        f1_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                                                trackers), desc='eval f1', total=len(trackers), ncols=100):
                f1_result.update(ret)
        benchmark.show_result(f1_result,
                              show_video_level=args.show_video_level)
        if args.vis:
            draw_f1(f1_result)
