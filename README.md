# Filter Combination Network

This repository contains code that implements filter combination learning.

## requirements

- python      : 3.6
- torch       : 1.5.0
- torchvision : 0.6.0

## Mnist

```
python mnist_main.py --save_dir [PATH]
```

## Cifar10

**vgg16**

```
python cifar10_main.py --model_name vgg16 -e conv -t normal -o normal --save_dir [PATH]
```

**resnet18**


```
python cifar10_main.py --model_name resnet18 -e conv -t normal -o normal --save_dir [PATH]
```

## Cifar100

**vgg16**

```
python cifar100_main.py --model_name vgg16 -e conv -t normal -o normal --save_dir [PATH]
```

**resnet18**

```
python cifar100_main.py --model_name resnet18 -e conv -t normal -o normal --save_dir [PATH]
```
