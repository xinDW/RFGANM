from config import config
from workflow import Dataset, GANConfig, GAN

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--begin_epoch', type=int, default=0, help='0 is there are no checkpoints saved')
    
    args = parser.parse_args()

    begin_epoch = args.begin_epoch
    
    dataset = Dataset(hr_training_path=config.TRAIN.hr_img_path, lr_training_path=config.TRAIN.lr_img_path, lr_valid_path=config.VALID.lr_img_path, mode='train');
    mConfig = GANConfig()
    gan = GAN(dataset, mConfig)
    gan.train(begin_epoch)
        
    