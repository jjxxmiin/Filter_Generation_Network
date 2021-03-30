# Filter Generation Network

This repository contains code that implements filter combination learning.

## requirements

- python      : 3.6
- torch       : 1.5.0
- torchvision : 0.6.0

## Mnist

```shell script
python mnist_main.py --save_dir [PATH]
```

## Cifar10

**vgg16**

```shell script
python cifar10_main.py --model_name vgg16 -e conv -t normal -o normal --save_dir [PATH]
```

**resnet18**

```shell script
python cifar10_main.py --model_name resnet18 -e conv -t normal -o normal --save_dir [PATH]
```

## Cifar100

**vgg16**

```shell script
python cifar100_main.py --model_name vgg16 -e conv -t normal -o normal --save_dir [PATH]
```

**resnet18**

```shell script
python cifar100_main.py --model_name resnet18 -e conv -t normal -o normal --save_dir [PATH]
```

## Rebuild

- VGG16 example
- GF Layer -> Conv Layer

```python
for i, (name, module)in enumerate(model.features.named_modules()):
    if isinstance(module, GFLayer):
        current_layer += 1

        in_channels = module.in_ch
        out_channels = module.out_ch
        groups = module.groups
        stride = module.stride
        padding = module.padding

        if current_layer <= 8:
            f = middle_filters
        else:
            f = last_filters

        new_weights = f.view(1, 1, 3, 3, 3) * \
            module.weights.view(out_channels, in_channels // groups, 3, 1, 1).repeat(1, 1, 1, 3, 3)

        new_weights = new_weights.sum(2)

        new_conv = torch.nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   stride=stride,
                                   padding=padding,
                                   groups=groups,
                                   bias=(module.bias is not None)).to(device)

        new_conv.weight.data = new_weights
        model.features[i-1] = new_conv
```

## Citation

```
@article{jeong2021filter,
  title={Filter combination learning for CNN model compression},
  author={Jeong, Jaemin and Cho, Ji-Ho and Lee, Jeong-Gun},
  journal={ICT Express},
  volume={7},
  number={1},
  pages={5--9},
  year={2021},
  publisher={Elsevier}
}
```
