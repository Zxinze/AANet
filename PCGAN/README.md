
# PCGAN_Model

## Prerequisites
- Linux 
- NVIDIA GPU + CUDA CuDNN 
- PyTorch >=1.1.0

## Train
You can run our training code in gvien examples of dataset folder by following instruction.
```bash
python train.py  --dataroot=./dataset --batch_size=1
```
Models are saved to `./checkpoints/`.

See `opt` in files(base_options.py and train_options.py) for additional training options.

## Test
You can test our pretrained model in gvien examples of dataset folder by following instruction.
```bash
python test.py --dataroot=./dataset --epoch=200 --data_type=short
```
'--epoch=200' means loading the 200th epoch saved model.

'--data_type=short or long', 'short' means testing imges of short exposure time, while 'long' means testing imges of long exposure time (saturation images).

The pretrained model can be downloaded at [Baidu Netdisk](https://pan.baidu.com/s/1ET_HHdvNyI8l-gjHg31ceg), password: `tr2n`.

Testing results are saved in `./result/`， and each result inculdes 8 images, I<sub>A</sub>, I<sub>M</sub>, I<sub>G</sub>, I<sub>comp</sub>, I<sub>GT</sub>, I<sub>SAT</sub>, M(I<sub>G</sub>), M(I<sub>GT</sub>), Where the mean of these tokens are in our paper.

See `opt` in files (base_options.py and test_options.py) for additional testing options.

## Dataset
Several samples of our dataset are in `./dataset`. Each sample incules 3 images, I<sub>SAT</sub>, I<sub>GT</sub>, I<sub>M</sub>. The size of training image is 350 x 350, but they are randomly crop to 256 x 256 in training process. Testing image is 256 x 256. 

## Acknowledgments
Code borrows heavily from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [partialconv](https://github.com/NVIDIA/partialconv). 

