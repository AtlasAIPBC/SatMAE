python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=1 --master_port=1234 main_finetune.py \
    --output_dir /home/ada/satmae/multispectral/evaluation/finetune_vit_large \
    --log_dir /home/ada/satmae/multispectral/evaluation/finetune_vit_large \
    --device cpu \
    --batch_size 16 \
    --patch_size 8 \
    --model vit_large_patch16 \
    --model_type group_c \
    --input_size 96 \
    --resume https://zenodo.org/record/7338613/files/finetune-vit-large-e7.pth \
    --dist_eval --eval --num_workers 8 --dataset sentinel \
    --train_path /home/ada/satmae/multispectral/data/fmow-sentinel/train.csv \
    --test_path /home/ada/satmae/multispectral/data/fmow-sentinel/val.csv \
    --nb_classes 62
