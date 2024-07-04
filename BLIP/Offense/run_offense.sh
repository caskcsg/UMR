CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=25644 --use_env Offense.py \
--config ./configs/Offense.yaml \
--output_dir /mnt/raid/yangcp/output/blip/cen/d




