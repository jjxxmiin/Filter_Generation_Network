## Dataset Structure

```
Prune_QTorch
  | datasets
     | cifar10 | train | airplane | .jpg ...
     |         |       | dog      | .jpg ...
     |         | test  
     |
     | tiny_imagenet  | train | Class1   | .jpg ...
                      |       | Class2   | .jpg ...
                      | test
```

- cifar10 : pytorch loader class
- tiny_imagenet : pytorch image folder module

Will be integrated image folder module..

## Download

- tiny imagenet download : [https://tiny-imagenet.herokuapp.com/](https://tiny-imagenet.herokuapp.com/)

```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

- cifar10 download       : [https://www.kaggle.com/swaroopkml/cifar10-pngs-in-folders](https://www.kaggle.com/swaroopkml/cifar10-pngs-in-folders) 

## Dataset info

Dataset infomation

#### tiny imagenet

- class : 200
- train : 500 per class
- test  : 50 per class
- valid : 50 per class 

#### cifar10 

- train : 5000 per class
- test  : 1000 per class
