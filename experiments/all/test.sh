# test otb100 dataset
nohup mpiexec --allow-run-as-root -n 1 \
python /data/SiamBC/tracking/test_epochs.py \
--start_epoch 50 \
--end_epoch 50 \
--gpu_nums 1 \
--threads 1 \
--dataset OTB100 \
--align True \
--type all >>test_otb.log 2>&1 &

tail test_otb.log -f