python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=1 --master_port=1234 /home/ada/satmae/SatMAE/main_finetune.py \
    --batch_size 16 \
    --output_dir /home/ada/satmae/temporal/evaluation/single_image_classification/pretrain \
    --log_dir /home/ada/satmae/temporal/evaluation/single_image_classification/pretrain \
    --device cpu \
    --model vit_large_patch16 \
    --resume https://zenodo.org/record/7369797/files/fmow_pretrain.pth \
    --dist_eval --eval --num_workers 8 --dataset rgb \
    --train_path /home/ada/satmae/temporal/preprocessed/fmow/train/train_62classes.csv \
    --test_path /home/ada/satmae/temporal/preprocessed/fmow/train/val_62classes.csv \
    --nb_classes 1000
