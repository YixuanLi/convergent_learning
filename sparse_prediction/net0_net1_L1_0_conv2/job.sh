#! /bin/bash

#SBATCH -A EvolvingAI
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jason@yosinski.com

echo "Command run: $0 $@"

python /home/jyosinsk/s/caffe/experiments/local-distrib/lasso.py \
    --net0_proto /project/EvolvingAI/jyosinsk/moran_results/150506_new_nets/150506_224456_2b38b3e_priv-ld_caffenet_nogrp0/deploy.prototxt \
    --net0_model /project/EvolvingAI/jyosinsk/moran_results/150506_new_nets/150506_224456_2b38b3e_priv-ld_caffenet_nogrp0/train_iter_450000.caffemodel \
    --net1_proto /project/EvolvingAI/jyosinsk/moran_results/150506_new_nets/150506_224502_2b38b3e_priv-ld_caffenet_nogrp1/deploy.prototxt\
    --net1_model /project/EvolvingAI/jyosinsk/moran_results/150506_new_nets/150506_224502_2b38b3e_priv-ld_caffenet_nogrp1/train_iter_450000.caffemodel \
    --stitch_proto solver.prototxt \
    --train_datadir /home/jyosinsk/imagenet2012/train/ \
    --train_filelist /home/jyosinsk/s/caffe/data/ilsvrc12/data/whole_train/shuffled_files.txt \
    --val_datadir /home/jyosinsk/imagenet2012/val/ \
    --val_filelist /home/jyosinsk/s/caffe/data/ilsvrc12/data/whole_valid/shuffled_files.txt \
    --layer_name conv2 \
    --maxiter 4501 \
    --gpu
