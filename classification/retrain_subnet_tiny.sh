# python train.py \
bash ./distributed_train_normal.sh 8 \
--aa 'rand-m9-mstd0.5' \
--amp \
--apex-amp \
--aug-splits 0 \
-b 128 \
--cutmix 1.0 \
-data_dir '/home/tiger/datasets/imagenet/' \
--decay-epochs 2.4 \
--decay-rate 0.973 \
--dist-bn '' \
-distributed \
--drop 0.2 \
--drop-path 0.1 \
--epochs 600 \
--img-size 224 \
--lr 0.064 \
--lr-noise 0.42 0.9 \
--min-lr 1e-05 \
--mixup 0.2 \
--model 'hess_base' \
--model-ema \
--model-ema-decay 0.9999 \
--num-classes 1000 \
--opt 'rmsproptf' \
--opt-eps 0.001 \
--output 'output/' \
--ratio 0.75 1.3333333333333333 \
--reprob 0.2 \
--sched 'step' \
--warmup-lr 1e-06 \
--weight-decay 1e-05 \
--workers 8 \
--warmup-epochs 3 \
--experiment 'hess' \
--log-wandb \

bash ~/release_watchdog.sh
# --clip-grad 1.0 \
# --sync-bn \
# --resume 'output/bignas_bs256_warmup20_epoch_900_with_two_subnet/last.pth.tar' \
# --drop-connect 0.2 \ # 已废弃
# --num-gpu 1 \
# --world-size 8 \
# --bn-eps 默认None \
# --bn-momentum 默认None \
# --bn_tf 默认False \
# --channels-last 默认为False \
# --clip-grad 默认None \
# --color-jitter 默认0.4 \
# --cooldown-epochs 默认10 \
# --crop-pct 默认None \
# --cutmix-minmax 默认None \
# --drop-block 默认None \
# --eval-checkpoint 没找到该属性 \
# --eval-metric 默认为'top1' \
# --gp 默认None \
# --hflip 默认0.5 \
# --initial-checkpoint 默认'' \
# --interpolation', 默认'' \
# --jsd_loss 默认False \ 
# --local_rank 默认0 \
# --log-interval', 默认50 \
# --lr-cycle-limit 默认1 \
# --lr-cycle-mul 默认1.0 \
# --lr_noise_pct 默认0.67 \
# --lr-noise-std 默认1.0 \
# --mean 默认None
# --mixup-mode 默认为'batch' \
# --mixup-off-epoch 默认0 \
# --mixup-prob 默认1.0 \
# --mixup-switch-prob 默认0.5 \
# --model-ema-force-cpu 默认False \
# --momentum 默认0.9 \
# --native-amp 默认False \
# --no-aug 默认False \
# --no-prefetcher 默认False \
# --no-resume-opt 默认False \
# --opt-betas 默认False \
# --patience-epochs 默认10 \
# --pin-mem 默认False \
# --pretrained 默认False \
# --recount 默认1 \
# --recovery-interval 默认0 \
# --remode 默认'pixel' \
# --resplit 默认False \
# --resume 默认'' \
# --save-images 默认False \
# --scale 默认[0.08, 1.0] \
# --seed 默认42 \
# --smoothing 默认0.1 \
# --split-bn 默认False \
# --start-epoch 默认None \
# --std 默认None \
# --train-interpolation 默认'random' \
# --tta 默认0 \
# --use-multi-epochs-loader 默认False \
# --validation-batch-size-multiplier 默认为1 \
# --vflip 默认0.0 \
# --warmup-epochs 默认3 \
