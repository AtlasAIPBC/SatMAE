python -m torch.distributed.launch --nproc_per_node=1 \
    --nnodes=1 --master_port=1234 main_finetune.py \
    --output_dir /home/ada/satmae/temporal/evaluation/single_image_classification/pretrain \
    --log_dir /home/ada/satmae/temporal/evaluation/single_image_classification/pretrain \
    --device cuda \
    --batch_size 8 \
    --model vit_large_patch16 \
    --resume https://zenodo.org/record/7369797/files/fmow_pretrain.pth  \
    --dist_eval --eval --num_workers 8 --dataset rgb \
    --train_path /home/ada/satmae/temporal/preprocessed/fmow/train/train_62classes.csv \
    --test_path /home/ada/satmae/temporal/preprocessed/fmow/train/val_62classes.csv \
    --nb_classes 1000
