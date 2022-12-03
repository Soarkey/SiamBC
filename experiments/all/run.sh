# run shell
nohup python /data/SiamBC/tracking/train.py \
  --cfg train.yaml >> train.log 2>&1 &

tail train.log -f