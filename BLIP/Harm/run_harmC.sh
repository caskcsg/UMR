CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=25641 --use_env HarmC.py \
--config ./configs/HarmC.yaml \
--output_dir /mnt/raid/yangcp/output/blip/cen/d




