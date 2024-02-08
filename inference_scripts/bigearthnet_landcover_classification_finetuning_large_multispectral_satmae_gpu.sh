
python -m torch.distributed.launch --nproc_per_node=1 \
    --nnodes=1 --master_port=1235 /home/ada/satmae/SatMAE/main_finetune.py \
    --device cuda \
    --batch_size 4 --accum_iter 16 --blr 0.0002 --lr 0.001 \
    --epochs 10 --num_workers 16 \
    --input_size 96 --patch_size 8  \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --model_type group_c \
    --dataset_type bigearthnet \
    --train_path /home/ada/satmae/other_data/bigearthnet/train_multi_band.csv \
    --test_path /home/ada/satmae/other_data/bigearthnet/val_multi_band.csv \
    --output_dir /home/ada/satmae/other_data/bigearthnet/evaluation/multispectral \
    --log_dir /home/ada/satmae/other_data/bigearthnet/evaluation/multispectral \
    --finetune /home/ada/satmae/other_data/bigearthnet/evaluation/multispectral/checkpoint-0_1.pth  \
    --nb_classes 19
