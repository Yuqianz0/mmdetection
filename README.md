
## Usage

### Requirements
* Python 3.6+
* Pytorch 1.5.0
* mmcv-full 1.0.5

### Datasets
    data
      ├── coco
      |   ├── annotations
      │   │   │   ├── instances_train2017.json
      │   │   │   ├── instances_val2017.json
      │   ├── train2017
      │   │   ├── 000000004134.png
      │   │   ├── 000000031817.png
      │   │   ├── ......
      │   ├── val2017
      │   ├── test2017


### Training
```
./scripts/dist_train.sh ./configs/refinemask/coco/r50-refinemask-1x-msdfpn.py 8 work_dirs/r50-refinemask-1x-msdfpn
```

### Inference
```
./scripts/dist_test.sh ./configs/refinemask/coco/r50-refinemask-1x-msdfpn.py 8 work_dirs/r50-refinemask-1x-msdfpn
```

