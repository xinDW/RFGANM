from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

config.TRAIN.hr_size = 384
config.TRAIN.lr_size = 96
config.TRAIN.n_channels = 3
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

#config.TRAIN.hr_img_path = 'path-to-HRs'
#config.TRAIN.lr_img_path = 'path-to-LRs'
config.TRAIN.hr_img_path = 'data/py_test/hr/'
config.TRAIN.lr_img_path = 'data/py_test/lr/'

config.VALID = edict()

## test set location
#config.VALID.lr_img_path = 'path-to-test-lrs'
config.VALID.lr_img_path = 'data/py_test/lr/'




