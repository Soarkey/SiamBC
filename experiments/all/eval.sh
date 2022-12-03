nohup python /data/SiamBC/lib/eval_toolkit/bin/eval.py \
--dataset_dir /data/SiamBC/dataset/OTB100 \
--dataset OTB100 \
--tracker_result_dir ./result/OTB100 \
--trackers 50 \
>>eval_otb.log 2>&1 &

tail eval_otb.log -f