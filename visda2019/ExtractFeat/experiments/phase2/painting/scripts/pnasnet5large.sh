GPUID=3
RESUME=18
PHASE=2

DOMAIN=painting
NET=pnasnet5large
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py --folder ./experiments/phase$PHASE/${DOMAIN}/$NET/${NET}_real --resume $RESUME
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py --folder ./experiments/phase$PHASE/${DOMAIN}/$NET/${NET}_quickdraw --resume $RESUME
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py --folder ./experiments/phase$PHASE/${DOMAIN}/$NET/${NET}_infograph --resume $RESUME
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py --folder ./experiments/phase$PHASE/${DOMAIN}/$NET/${NET}_sketch --resume $RESUME
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py --folder ./experiments/phase$PHASE/${DOMAIN}/$NET/${NET}_${DOMAIN} --resume $RESUME

CUDA_VISIBLE_DEVICES=$GPUID python3 main.py --folder ./experiments/phase$PHASE/${DOMAIN}/$NET/${NET}_real --resume $RESUME --mirror
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py --folder ./experiments/phase$PHASE/${DOMAIN}/$NET/${NET}_quickdraw --resume $RESUME --mirror
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py --folder ./experiments/phase$PHASE/${DOMAIN}/$NET/${NET}_infograph --resume $RESUME --mirror
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py --folder ./experiments/phase$PHASE/${DOMAIN}/$NET/${NET}_sketch --resume $RESUME --mirror
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py --folder ./experiments/phase$PHASE/${DOMAIN}/$NET/${NET}_${DOMAIN} --resume $RESUME --mirror
