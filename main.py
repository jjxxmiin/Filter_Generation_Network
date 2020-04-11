import os

os.system("source activate pytorch_p36")
os.system("CUDA_VISIBLE_DEVICES=1 python ./benchmark/l1_norm/prune.py")
