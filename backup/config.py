from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 100
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 2000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location

#config.TRAIN.hr_img_path = 'data/train/huvec/rgb384_sigma0.800000by4var0.000090_overlap0.0/hr/'
#config.TRAIN.lr_img_path = 'data/train/huvec/rgb384_sigma0.800000by4var0.000090_overlap0.0/lr/'
config.TRAIN.hr_img_path = 'data/train/pathology_section/10X-2.5X/7-1/hr/'
config.TRAIN.lr_img_path = 'data/train/pathology_section/10X-2.5X/7-1/lr/'

config.VALID = edict()
## test set location

config.VALID.lr_img_path = 'data/lr/'
#config.VALID.lr_img_path = 'data/mouse_brain/20180118/1.6X-simu-selected/'
#config.VALID.lr_img_path = 'data/pathological section/6-1/g1/2.5x/cropped96_overlap0.2/'


config.TRAIN.hr_size = 384
config.TRAIN.lr_size = 96
def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================/n")
        f.write(json.dumps(cfg, indent=4))
        f.write("/n================================================/n")
