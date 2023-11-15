python -m torch.distributed.launch --nproc_per_node=1 \
    --nnodes=1 --master_port=1234 /home/ada/satmae/SatMAE/main_finetune.py \
    --output_dir /home/ada/satmae/other_data/spacenet_v1/evaluation/multispectral_finetune_large \
    --log_dir /home/ada/satmae/other_data/spacenet_v1/evaluation/multispectral_finetune_large \
    --device cuda \
    --batch_size 8 \
    --patch_size 8 \
    --model vit_large_patch16 \
    --model_type group_c \
    --input_size 96 \
    --resume /home/ada/satmae/multispectral/checkpoints/finetune-vit-large-e7.pth  \
    --dist_eval --eval --num_workers 8 --dataset spacenet \
    --train_path /home/ada/satmae/other_data/spacenet_v1/new/train/summarydata/8band_train.csv \
    --test_path /home/ada/satmae/other_data/spacenet_v1/new/train/summarydata/8band_val.csv \
    --nb_classes 62

