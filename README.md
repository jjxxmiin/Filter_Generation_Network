# Pruning

# DATASETS

- imagenet

```shell

- datasets
    - imagenet - train
               - val
               - valprep.sh
- Prune_QTorch
```

valprep.sh [Download](https://github.com/pytorch/examples/tree/master/imagenet)

# Benchmark

## Origin

- More acc   : [https://pytorch.org/docs/stable/torchvision/models.html](https://pytorch.org/docs/stable/torchvision/models.html)
- More flops : [https://github.com/Lyken17/pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)
- pytorch pretrained model

|Network|Top-1 error|Top-5 error|Param(M)|MACs(G)|
|------|---|---|---|---|
|VGG-11|30.98|11.37|132.86|7.74|
|VGG-16|28.41|9.62|138.36|15.61|
|VGG-11 with batch normalization|29.62|10.19|132.87|7.77|
|VGG-13 with batch normalization|28.45|9.63|133.05|11.49|
|VGG-16 with batch normalization|26.63|8.50|138.37|15.66|
|ResNet-34|26.70|8.58|21.80|3.68|
|ResNet-50|23.85|7.13|25.56|4.14|

# REFERENCE
- [https://github.com/pytorch/examples/tree/master/imagenet](https://github.com/pytorch/examples/tree/master/imagenet)
- [https://pytorch.org/docs/stable/torchvision/models.html](https://pytorch.org/docs/stable/torchvision/models.html)
- [https://github.com/Lyken17/pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)
