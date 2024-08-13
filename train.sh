#CUDA_VISIBLE_DEVICES=0 python main.py --

CUDA_VISIBLE_DEVICES=2 python -u main.py --dataset 'SetFit/sst5' --model 'bert-base-uncased' --alg 'routing_adapter' --device '0' --adapter 'routing_adapter' --batch_size 32 --epochs 8
    