python -m torch.distributed.launch --nproc_per_node=1 \
    --nnodes=1 --master_port=1234 /home/ada/satmae/SatMAE/main_finetune.py \
    --output_dir /home/ada/satmae/other_data/naip/evaluation \
    --log_dir /home/ada/satmae/other_data/naip/evaluation \
    --batch_size 16 --accum_iter 4 \
    --model vit_large_patch16 --epochs 20 --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 \
    --mixup 0.8 --cutmix 1.0 \
    --finetune /home/ada/satmae/temporal/fmow_pretrain.pth \
    --dist_eval --num_workers 8 --dataset naip \
