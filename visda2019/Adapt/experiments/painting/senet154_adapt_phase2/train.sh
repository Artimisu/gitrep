cp ./experiments/painting/senet154/snapshot/* ./experiments/painting/senet154_adapt_phase2/snapshot
CUDA_VISIBLE_DEVICES=3,2,1,0 python3 -m torch.distributed.launch --nproc_per_node=4 main.py --folder ./experiments/painting/senet154_adapt_phase2 --resume 4
