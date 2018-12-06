## SRGAN for Microscopy imaging

### Environment
The script is tested on:

- windows 7 (tensorflow 1.8.0, CUDA 9.0, python 3.5)
- ubuntu14.04 (tensorflow 1.9.0, CUDA9.0, python 2.7)

An old-version [TensorLayer](http://tensorlayer.readthedocs.io/en/latest/) is used (self contained). Update TensorLayer at your own risk.


### Prepare Data and Pre-trained VGG

- 1. Download the pretrained VGG19 model in [here](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs)
- 2. High resolution(HR) and low resolution(LR) image pairs are needed to trian the GAN. 


### Train the network

- Set your image folder in `config.py`

```python
config.TRAIN.hr_img_path = "hr_image_folder/"
config.TRAIN.lr_img_path = "lr_image_folder/"
```
- Start training.

```bash
python trian.py
```

### Inference after well-trained

- set path for LR imaegs that you want to reconstruct.

```python 
config.VALID.lr_img_path = 'lr_image_folder'

- start inferencing with: 
```
```bash
python inference.py --begin_epoch=<your_checkpoint_file_number> 
```


### Reference
* [1] [High-throughput, high-resolution registration-free generated adversarial network microscopy](https://arxiv.org/abs/1801.07330)
* [2] [SRGAN](https://github.com/tensorlayer/srgan)

### License

- For academic and non-commercial use only.
