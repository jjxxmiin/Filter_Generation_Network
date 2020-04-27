# Network Trimming(Apoz)

- Paper : [Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures](https://arxiv.org/abs/1607.03250)

## Benchmark

- Simple
- pruning layer : `Conv 5-3`, `FC 6`

### Apoz

### Pruning

```shell script
Before Pruning
Acc@1: 71.59 
Acc@5: 90.38

After Pruning
Acc@1: 70.37
Acc@5: 89.76
```

### Fine tune

```shell script
Before Fine tune
Acc@1: 70.37
Acc@5: 89.76

After Fine tune

1 Epoch
Acc@1 70.57
Acc@5 89.88
```

## Reference
- [Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures](https://arxiv.org/abs/1607.03250)
