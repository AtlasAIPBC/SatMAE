python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=1 --master_port=1234 main_pretrain.py \
    --output_dir /home/ada/satmae/temporal/evaluation/single_image_classification/pretrain \
    --log_dir /home/ada/satmae/temporal/evaluation/single_image_classification/pretrain \
    --batch_size 16 \
    --model mae_vit_large_patch16 \
    --resume https://zenodo.org/record/7369797/files/fmow_pretrain.pth \
    --num_workers 8 --dataset rgb \
    --train_path /home/ada/satmae/temporal/preprocessed/fmow/train/val_62classes.csv \
    --device cpu
