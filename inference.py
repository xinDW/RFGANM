from config import config
from workflow import Dataset, GANConfig, GAN
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--begin_epoch', type=int, default=0, help='0 is there are no checkpoints saved')
    parser.add_argument('--end_epoch', type=int, default=0, help='')
    parser.add_argument('--interval', type=int, default=10, help='')
    args = parser.parse_args()

    begin_epoch = args.begin_epoch
    end_epoch = args.end_epoch
    interval = args.interval
    
    dataset = Dataset(hr_training_path=config.TRAIN.hr_img_path, lr_training_path=config.TRAIN.lr_img_path, lr_valid_path=config.VALID.lr_img_path, mode='inference');
    mConfig = GANConfig()
    gan = GAN(dataset, mConfig)
    
    if end_epoch != 0:
        if end_epoch > begin_epoch:
            gan.inference(begin_epoch, end_epoch, interval)
        else:
            raise Exception("end_epoch is smaller than begin_epoch")
    else:
        gan.inference(begin_epoch, begin_epoch + 1, interval)
        