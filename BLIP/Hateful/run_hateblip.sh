CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=25644 --use_env Hateful_blip.py \
--config ./configs/Hateful_blip.yaml \
--output_dir /mnt/raid/yangcp/output/blip/cen/tesdt




