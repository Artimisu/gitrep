cp ./experiments/clipart/inceptionresnetv2/snapshot/* ./experiments/clipart/inceptionresnetv2_adapt_phase3/snapshot
CUDA_VISIBLE_DEVICES=3,2,1,0 python3 -m torch.distributed.launch --nproc_per_node=4 main.py --folder ./experiments/clipart/inceptionresnetv2_adapt_phase3 --resume 4
