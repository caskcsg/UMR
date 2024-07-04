CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=25649 --use_env Offense.py \
--config ./configs/Offense.yaml \
--output_dir /mnt/raid/yangcp/output/albef/cen/d \
--checkpoint /mnt/raid/yangcp/checkpoint/ALBEF/ALBEF.pth 




