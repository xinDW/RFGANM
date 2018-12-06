import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config

    
class GANConfig:
    def __init__(self):
        ## HYPER-PARAMETERS
        self.hr_size = config.TRAIN.hr_size
        self.lr_size = config.TRAIN.lr_size
        self.n_channels = config.TRAIN.n_channels
        self.batch_size = config.TRAIN.batch_size
        self.lr_init = config.TRAIN.lr_init
        self.beta1 = config.TRAIN.beta1

        self.n_epoch_init = config.TRAIN.n_epoch_init

        self.n_epoch = config.TRAIN.n_epoch
        self.lr_decay = config.TRAIN.lr_decay
        self.decay_every = config.TRAIN.decay_every
        
        ## folders to save result images and trained model
        self.save_dir_ginit = "samples/init"
        self.save_dir_gan = "samples/gan"      
        self.checkpoint_dir = "checkpoint"
        self.save_dir_inference = "samples/inference"
        
        tl.files.exists_or_mkdir(self.save_dir_ginit)
        tl.files.exists_or_mkdir(self.save_dir_gan)
        tl.files.exists_or_mkdir(self.checkpoint_dir)
        tl.files.exists_or_mkdir(self.save_dir_inference)
    
    
class Dataset:
    def __init__(self, hr_training_path, lr_training_path, lr_valid_path, mode='train'):
        self.hr_training_path = hr_training_path
        self.lr_training_path = lr_training_path
        self.lr_valid_path = lr_valid_path
        
        self.hr_training_img_list = None
        self.lr_training_img_list = None
        self.lr_valid_img_list = None        
        if mode == 'train':
            self.hr_training_img_list = sorted(tl.files.load_file_list(path=hr_training_path, regx='.*.png', printable=False))
            self.lr_training_img_list = sorted(tl.files.load_file_list(path=lr_training_path, regx='.*.png', printable=False))
        if mode == 'inference':
            self.lr_valid_img_list = sorted(tl.files.load_file_list(path=lr_valid_path, regx='.*.png', printable=False))
    
    def read_all_imgs(self):
        hr_training_imgs = read_imgs(self.hr_training_img_list, path=self.hr_training_path)
        lr_training_imgs = read_imgs(self.lr_training_img_list, path=self.lr_training_path)
        lr_valid_imgs = read_imgs(self.lr_valid_img_list, path=self.lr_valid_path)
        
        return hr_training_imgs, lr_training_imgs, lr_valid_imgs
        
class GAN:
    def __init__(self, dataset, mConfig):
        self.dataset = dataset
        self.mConfig = mConfig
        
        self.hr_size = mConfig.hr_size
        self.lr_size = mConfig.lr_size
        self.n_channels = mConfig.n_channels 
        
        self.hr_training_imgs, self.lr_training_imgs, self.lr_valid_imgs = dataset.read_all_imgs()
        
        configProto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        configProto.gpu_options.allow_growth=True
        self.configProto = configProto
        
    def train(self, begin_epoch):
        batch_size = self.mConfig.batch_size
        ckpt_dir = self.mConfig.checkpoint_dir
        
        t_image = tf.placeholder('float32', [batch_size, self.lr_size, self.lr_size, self.n_channels], name='t_image_input_to_generator')
        t_target_image = tf.placeholder('float32', [batch_size, self.hr_size, self.hr_size, self.n_channels], name='t_target_image')

        net_g = SRGAN_g(t_image, is_train=True, reuse=False)
        net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
        _,     logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

        net_g.print_params(False)
        net_d.print_params(False)

        ## vgg. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
        t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0, align_corners=False) # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
        t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False) # resize_generate_image_for_vgg

        net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224+1)/2, reuse=False)
        _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224+1)/2, reuse=True)

        ## test 
        net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

        # ###========================== DEFINE TRAIN OPS ==========================###
        d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
        d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
        d_loss = d_loss1 + d_loss2

        g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
        mse_loss = tl.cost.mean_squared_error(net_g.outputs , t_target_image, is_mean=True)
        vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

        g_loss = mse_loss + vgg_loss + g_gan_loss

        g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
        d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

        with tf.variable_scope('learning_rate'):
            lr_v = tf.Variable(self.mConfig.lr_init, trainable=False)
        ## Pretrain
        g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=self.mConfig.beta1).minimize(mse_loss, var_list=g_vars)
        ## SRGAN
        g_optim = tf.train.AdamOptimizer(lr_v, beta1=self.mConfig.beta1).minimize(g_loss, var_list=g_vars)
        d_optim = tf.train.AdamOptimizer(lr_v, beta1=self.mConfig.beta1).minimize(d_loss, var_list=d_vars)
        
        sess = tf.Session(config=self.configProto)
        tl.layers.initialize_global_variables(sess)
        if begin_epoch != 0:
            if tl.files.load_and_assign_npz(sess=sess, name=ckpt_dir+'/g_epoch{}.npz'.format(begin_epoch), network=net_g) is False:
                raise Exception(ckpt_dir + '/g_epoch{}.npz doesn\'t exist '.format(begin_epoch) )
                #tl.files.load_and_assign_npz(sess=sess, name=self.mConfig.checkpoint_dir+'/g_init.npz', network=net_g)
            if tl.files.load_and_assign_npz(sess=sess, name=self.mConfig.checkpoint_dir+'/d_epoch{}.npz'.format(begin_epoch), network=net_d) is False:
                raise Exception(ckpt_dir + '/d_epoch{}.npz doesn\'t exist '.format(begin_epoch) )
        
        ###============================= LOAD VGG ===============================###
        vgg19_npy_path = "vgg19.npy"
        if not os.path.isfile(vgg19_npy_path):
            print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
            exit()
        npz = np.load(vgg19_npy_path, encoding='latin1').item()

        params = []
        for val in sorted( npz.items() ):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            params.extend([W, b])
        tl.files.assign_params(sess, params, net_vgg)
        # net_vgg.print_params(False)
        # net_vgg.print_layers()

        ###============================= TRAINING ===============================###
        ## use first `batch_size` of train set to have a quick test during training
        sample_imgs = self.hr_training_imgs[0 : batch_size]
        sample_lr_imgs = self.lr_training_imgs[0 : batch_size]
        
        sample_imgs_hr = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
        print('sample HR sub-image:',sample_imgs_hr.shape, sample_imgs_hr.min(), sample_imgs_hr.max())
        sample_imgs_lr = tl.prepro.threading_data(sample_lr_imgs, fn=scale_fn)
        print('sample LR sub-image:', sample_imgs_lr.shape, sample_imgs_lr.min(), sample_imgs_lr.max())
        
        ni = int(np.sqrt(batch_size))
        ni = ni + 1 if ni * ni < batch_size else ni
        
        tl.vis.save_images(sample_imgs_lr, [ni, ni], self.mConfig.save_dir_ginit+'/_train_sample_lr.png')
        tl.vis.save_images(sample_imgs_hr, [ni, ni], self.mConfig.save_dir_ginit+'/_train_sample_hr.png')
        tl.vis.save_images(sample_imgs_lr, [ni, ni], self.mConfig.save_dir_gan+'/_train_sample_lr.png')
        tl.vis.save_images(sample_imgs_hr, [ni, ni], self.mConfig.save_dir_gan+'/_train_sample_hr.png')

        ###========================= initialize G ====================###
        ## fixed learning rate
        sess.run(tf.assign(lr_v, self.mConfig.lr_init))
        
        if begin_epoch == 0:
            n_epoch_init = self.mConfig.n_epoch_init
            for epoch in range(0, n_epoch_init+1):
                epoch_time = time.time()
                total_mse_loss, n_iter = 0, 0

                ## If your machine have enough memory, please pre-load the whole train set.
                for idx in range(batch_size, len(self.hr_training_imgs)-batch_size, batch_size):
                    step_time = time.time()
                    b_imgs_384 = tl.prepro.threading_data(
                            self.hr_training_imgs[idx : idx + batch_size],
                            fn=crop_sub_imgs_fn, is_random=False)
                    b_imgs_96 = tl.prepro.threading_data(self.lr_training_imgs[idx : idx + batch_size], fn=crop_sub_imgs_fn, is_random=False)
                    
                    #print(b_imgs_384.shape)
                    #print(b_imgs_96.shape)
                    
                    ## update G
                    errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_96, t_target_image: b_imgs_384})
                    print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
                    total_mse_loss += errM
                    n_iter += 1
                log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss/n_iter)
                print(log)

                ## quick evaluation on train set
                if (epoch != 0) and (epoch % 2 == 0):
                    out = sess.run(net_g_test.outputs, {t_image: sample_imgs_lr})
                    print("[*] save images")
                    tl.vis.save_images(out, [ni, ni], self.mConfig.save_dir_ginit+'/train_%d.png' % epoch)
                ## save model
                if (epoch != 0) and (epoch % 2 == 0):
                    tl.files.save_npz(net_g.all_params, name=self.mConfig.checkpoint_dir+'/g_init.npz', sess=sess)
                    
                   
        ###========================= train GAN (SRGAN) =========================###
        n_epoch = self.mConfig.n_epoch
        for epoch in range(begin_epoch+1, n_epoch+1):
            ## update learning rate
            if epoch !=0 and (epoch % decay_every == 0):
                new_lr_decay = lr_decay ** (epoch // decay_every)
                sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
                log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
                print(log)
            elif epoch == 0:
                sess.run(tf.assign(lr_v, lr_init))
                log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
                print(log)

            epoch_time = time.time()
            total_d_loss, total_g_loss, n_iter = 0, 0, 0

            for idx in range(batch_size, len(self.hr_training_imgs) - batch_size, batch_size):
                step_time = time.time()
                b_imgs_384 = tl.prepro.threading_data(
                        self.hr_training_imgs[idx : idx + batch_size],
                        fn=crop_sub_imgs_fn, is_random=False)
                b_imgs_96 = tl.prepro.threading_data(self.lr_training_imgs[idx : idx + batch_size], fn=crop_sub_imgs_fn, is_random=False)
                ## update D
                errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})
                ## update G
                errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})
                print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" % (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
                total_d_loss += errD
                total_g_loss += errG
                n_iter += 1

            log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss/n_iter, total_g_loss/n_iter)
            print(log)

            ## quick evaluation on train set
            if (epoch != 0) and (epoch % 10 == 0):
                out = sess.run(net_g_test.outputs, {t_image: sample_imgs_lr})#; print('gen sub-image:', out.shape, out.min(), out.max())
                print("[*] save images")
                tl.vis.save_images(out, [ni, ni], self.mConfig.save_dir_gan+'/train_%d.png' % epoch)

            ## save model
            if (epoch != 0) and (epoch % 10 == 0):
                tl.files.save_npz(net_g.all_params, name=self.mConfig.checkpoint_dir+'/g_epoch{}.npz'.format(epoch), sess=sess)
                tl.files.save_npz(net_d.all_params, name=self.mConfig.checkpoint_dir+'/d_epoch{}.npz'.format(epoch), sess=sess)
   
    def inference(self, begin_epoch, end_epoch, interval):
        
        t_image = tf.placeholder('float32', [None, self.lr_size, self.lr_size, self.n_channels], name='input_image')

        net_g = SRGAN_g(t_image, is_train=False, reuse=False)

        ###========================== RESTORE G =============================###
        sess = tf.Session(config=self.configProto)
        tl.layers.initialize_global_variables(sess)
        
        if end_epoch < begin_epoch:
            raise Exception("end_epoch smaller than begin_epoch")
        else:
            for epoch in range(begin_epoch, end_epoch, interval):
                imid = 0 
                tl.files.load_and_assign_npz(sess=sess, name=self.mConfig.checkpoint_dir+'/g_epoch{}.npz'.format(epoch), network=net_g)
                
                # graph_def = tf.get_default_graph().as_graph_def()
                # tf.train.write_graph(graph_def, save_dir_inference, 'srgan_g.pb', as_text=False)
                
                ###======================= EVALUATION =============================###
                for valid_lr_img in self.lr_valid_imgs:
                  valid_lr_img = (valid_lr_img / 127.5) - 1.

                  start_time = time.time()
                  out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
                  print("took: %4.4fs" % (time.time() - start_time))
                  imid += 1
                  print("[*] epoch{} save images {}/{}".format(epoch, imid, len(self.lr_valid_imgs)))
                  tl.vis.save_image(out[0], self.mConfig.save_dir_inference+'/valid_gen{}_epoch{}.png'.format(imid, epoch))
                  
                  
