# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config, log_config

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))
while ni * ni < batch_size:
    ni = ni + 1
    
    
hr_size = config.TRAIN.hr_size
lr_size = config.TRAIN.lr_size

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def train(begin_epoch):
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    #valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    train_lr_imgs = read_all_imgs(train_lr_img_list, path=config.TRAIN.lr_img_path, n_threads=32)
        
   

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, lr_size, lr_size, 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [batch_size, hr_size, hr_size, 3], name='t_target_image')

    net_g = SRGAN_g(t_image, is_train=True, reuse=False)
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _,     logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    net_g.print_params(False)
    net_d.print_params(False)

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0, align_corners=False) # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False) # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224+1)/2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224+1)/2, reuse=True)

    ## test inference
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
        lr_v = tf.Variable(lr_init, trainable=False)
    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    ## SRGAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    ###========================== RESTORE MODEL =============================###
    configProto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    configProto.gpu_options.allow_growth=True
    sess = tf.Session(config=configProto)
    tl.layers.initialize_global_variables(sess)
    if begin_epoch != 0:
        if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}_epoch{}.npz'.format(tl.global_flag['mode'], begin_epoch), network=net_g) is False:
            tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/d_{}_epoch{}.npz'.format(tl.global_flag['mode'], begin_epoch), network=net_d)
    
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
    sample_imgs = train_hr_imgs[0:batch_size]
    sample_lr_imgs = train_lr_imgs[0:batch_size]
    
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    print('sample HR sub-image:',sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    sample_imgs_96 = tl.prepro.threading_data(sample_lr_imgs, fn=scale_fn)
    print('sample LR sub-image:', sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_ginit+'/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_ginit+'/_train_sample_384.png')
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_gan+'/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_gan+'/_train_sample_384.png')

    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    
    if begin_epoch == 0:
        for epoch in range(0, n_epoch_init+1):
            epoch_time = time.time()
            total_mse_loss, n_iter = 0, 0

            ## If your machine have enough memory, please pre-load the whole train set.
            for idx in range(batch_size, len(train_hr_imgs)-batch_size, batch_size):
                step_time = time.time()
                b_imgs_384 = tl.prepro.threading_data(
                        train_hr_imgs[idx : idx + batch_size],
                        fn=crop_sub_imgs_fn, is_random=False)
                b_imgs_96 = tl.prepro.threading_data(train_lr_imgs[idx : idx + batch_size], fn=crop_sub_imgs_fn, is_random=False)
                
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
            if (epoch != 0) and (epoch % 10 == 0):
                out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})#; print('gen sub-image:', out.shape, out.min(), out.max())
                print("[*] save images")
                tl.vis.save_images(out, [ni, ni], save_dir_ginit+'/train_%d.png' % epoch)
            ## save model
            if (epoch != 0) and (epoch % 10 == 0):
                tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)
                
               
    ###========================= train GAN (SRGAN) =========================###
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
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

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(batch_size, len(train_hr_imgs) - batch_size, batch_size):
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(
                    train_hr_imgs[idx : idx + batch_size],
                    fn=crop_sub_imgs_fn, is_random=False)
            b_imgs_96 = tl.prepro.threading_data(train_lr_imgs[idx : idx + batch_size], fn=crop_sub_imgs_fn, is_random=False)
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
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})#; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_gan+'/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}_epoch{}.npz'.format(tl.global_flag['mode'], epoch), sess=sess)
            tl.files.save_npz(net_d.all_params, name=checkpoint_dir+'/d_{}_epoch{}.npz'.format(tl.global_flag['mode'], epoch), sess=sess)
   
def evaluate(begin_epoch, end_epoch, interval):
    ## create folders to save result images
    
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=1)
    for im in valid_lr_imgs:
       print(im.shape)
       
    t_image = tf.placeholder('float32', [None, lr_size, lr_size, 3], name='input_image')
    # t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    
    if end_epoch < begin_epoch:
        raise Exception("end_epoch smaller than begin_epoch")
    else:
        for epoch in range(begin_epoch, end_epoch, interval):
            imid = 0 
            tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_srgan_epoch{}.npz'.format(epoch), network=net_g)
            
            graph_def = tf.get_default_graph().as_graph_def()
            #tf.train.write_graph(graph_def, save_dir, 'srgan_g.pb', as_text=False)
            
            ###======================= EVALUATION =============================###
            for valid_lr_img in valid_lr_imgs:
              valid_lr_img = (valid_lr_img / 127.5) - 1.
              # valid_lr_img = downsample_fn(valid_lr_img)

              start_time = time.time()
              out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
              print("took: %4.4fs" % (time.time() - start_time))
              imid += 1
              print("[*] epoch{} save images {}/{}".format(epoch, imid, len(valid_lr_imgs)))
              tl.vis.save_image(out[0], save_dir+'/valid_gen{}_epoch{}.png'.format(imid, epoch))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, eval')
    parser.add_argument('--begin_epoch', type=int, default=0, help='0 is there are no checkpoints saved')
    parser.add_argument('--end_epoch', type=int, default=0, help='')
    parser.add_argument('--interval', type=int, default=10, help='')
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    begin_epoch = args.begin_epoch
    
    if tl.global_flag['mode'] == 'srgan':
        train(begin_epoch)
    elif tl.global_flag['mode'] == 'eval':
        end_epoch = args.end_epoch
        interval = args.interval
        if end_epoch != 0:
            if end_epoch > begin_epoch:
                evaluate(begin_epoch, end_epoch, interval)
            else:
                raise Exception("end_epoch is smaller than begin_epoch")
        else:
            evaluate(begin_epoch, begin_epoch + 1, interval)
    else:
        raise Exception("Unknow --mode")
