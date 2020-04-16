# L1 Norm Pruning

- Paper : [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)

## Benchmark

- Simple
- 5 EPOCHS
- 10%

### Pruning

```shell script
Before Pruning
Acc@1: 72.06
Acc@5: 90.59
Param: 21797672

After Pruning
Acc@1: 61.13
Acc@5: 83.71
Param: 20151764
```

### Fine tune

```shell script
Before Fine tune
Acc@1: 61.13
Acc@5: 83.72
Param: 20151764

After Fine tune
Acc@1: 72.75
Acc@5: 91.10
Param: 20151764
```

## Reference
- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)
- [https://github.com/Eric-mingjie/rethinking-network-pruning](https://github.com/Eric-mingjie/rethinking-network-pruning)
