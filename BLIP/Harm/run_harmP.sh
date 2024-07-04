CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=25642 --use_env HarmP.py \
--config ./configs/HarmP.yaml \
--output_dir /mnt/raid/yangcp/output/blip/cen/d




