CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=25647 --use_env Hateful_v2.py \
--config ./configs/Hateful.yaml \
--output_dir /mnt/raid/yangcp/output/albef/cen/v2tes11 \
--checkpoint /mnt/raid/yangcp/checkpoint/ALBEF/ALBEF.pth 




