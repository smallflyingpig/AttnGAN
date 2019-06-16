from __future__ import print_function

from miscc.utils import mkdir_p, get_logger
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

from model import RNN_ENCODER, CNN_ENCODER
from audio_encoder import CNNRNN_Attn

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import logging
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter
import re


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/DAMSM/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--eval', action='store_true', default=False, help='enable eval mode')
    parser.add_argument('--update_interval', type=int, default=200, help="update interval" )
    parser.add_argument('--audio_flag', action='store_true', default=False, help="")
    args = parser.parse_args()
    return args


def train(dataloader, cnn_model, rnn_model, batch_size,
          labels, optimizer, epoch, ixtoword, image_dir, writer, logger, update_interval):
    cnn_model.train()
    rnn_model.train()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    count = (epoch) * len(dataloader)
    start_time = time.time()
    for step, data in enumerate(dataloader, 0):
        # print('step', step)
        rnn_model.zero_grad()
        cnn_model.zero_grad()

        imgs, captions, cap_lens, \
            class_ids, keys = prepare_data(data)
        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        words_features, sent_code = cnn_model(imgs[-1])
        # --> batch_size x nef x 17*17
        nef, att_sze = words_features.size(1), words_features.size(2)
        # words_features = words_features.view(batch_size, nef, -1)

        # hidden = rnn_model.init_hidden(batch_size)
        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_emb, sent_emb = rnn_model(captions, cap_lens)

        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                 cap_lens, class_ids, batch_size)
        w_total_loss0 += w_loss0.item()
        w_total_loss1 += w_loss1.item()
        loss = w_loss0 + w_loss1

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.item()
        s_total_loss1 += s_loss1.item()
        #
        loss.backward()
        #
        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(rnn_model.parameters(),
                                      cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()

        global_step = epoch * len(dataloader) + step
        writer.add_scalars(main_tag="batch_loss", tag_scalar_dict={
            "loss":loss.cpu().item(),
            "w_loss0":w_loss0.cpu().item(),
            "w_loss1":w_loss1.cpu().item(),
            "s_loss0":s_loss0.cpu().item(),
            "s_loss1":s_loss1.cpu().item()
        }, global_step=global_step)

        if step % update_interval == 0:
            count = epoch * len(dataloader) + step

            s_cur_loss0 = s_total_loss0 / update_interval
            s_cur_loss1 = s_total_loss1 / update_interval

            w_cur_loss0 = w_total_loss0 / update_interval
            w_cur_loss1 = w_total_loss1 / update_interval

            elapsed = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  's_loss {:6.4f} {:6.4f} | '
                  'w_loss {:6.4f} {:6.4f}'
                  .format(epoch, step, len(dataloader),
                          elapsed * 1000. / update_interval,
                          s_cur_loss0, s_cur_loss1,
                          w_cur_loss0, w_cur_loss1))
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            start_time = time.time()
            # attention Maps
        if global_step % (10*update_interval) == 0:
            img_set, _ = \
                build_super_images(imgs[-1][:,:3].cpu(), captions,
                                   ixtoword, attn_maps, att_sze)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/attention_maps%d.png' % (image_dir, count)
                im.save(fullpath)
                writer.add_image(tag="image_DAMSM", img_tensor=transforms.ToTensor()(im), global_step=count)
    return count


def evaluate(dataloader, cnn_model, rnn_model, batch_size, writer, count, ixtoword, labels, image_dir):
    cnn_model.eval()
    rnn_model.eval()
    s_total_loss = 0
    w_total_loss = 0
    for step, data in enumerate(dataloader, 0):
        real_imgs, captions, cap_lens, \
                class_ids, keys = prepare_data(data)

        words_features, sent_code = cnn_model(real_imgs[-1])
        # nef = words_features.size(1)
        # words_features = words_features.view(batch_size, nef, -1)
        nef, att_sze = words_features.size(1), words_features.size(2)

        # hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, cap_lens)

        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels,
                                            cap_lens, class_ids, batch_size)
        w_total_loss += (w_loss0 + w_loss1).data

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        s_total_loss += (s_loss0 + s_loss1).item()

        if step == 50:
            break

    s_cur_loss = s_total_loss / step
    w_cur_loss = w_total_loss / step

    writer.add_scalars( main_tag="eval_loss", tag_scalar_dict={
                        's_loss':s_cur_loss,
                        'w_loss':w_cur_loss
                    }, global_step=count
    )
    # save a image
    # attention Maps
    img_set, _ = \
        build_super_images(real_imgs[-1][:,:3].cpu(), captions,
                           ixtoword, attn, att_sze)
    if img_set is not None:
        im = Image.fromarray(img_set)
        fullpath = '%s/attention_maps_eval_%d.png' % (image_dir, count)
        im.save(fullpath)
        writer.add_image(tag="image_DAMSM_eval", img_tensor=transforms.ToTensor()(im), global_step=count)
    return s_cur_loss, w_cur_loss


def build_models(dataset, batch_size, audio_flag=False):
    # build model ############################################################
    if audio_flag:
        text_encoder = CNNRNN_Attn(40, nhidden=cfg.TEXT.EMBEDDING_DIM)
    else:
        text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM, nsent=cfg.TEXT.SENT_EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM, condition=cfg.TRAIN.MASK_COND, condition_channel=0)
    labels = torch.LongTensor(range(batch_size))
    start_epoch = 0
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        # istart = cfg.TRAIN.NET_E.rfind('encoder')
        # iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = re.match(r'.*_encoder(\d+).*', cfg.TRAIN.NET_E).group(1)
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)
    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()

    return text_encoder, image_encoder, labels, start_epoch


def main(args):
    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    
    # logging
    log_dir = os.path.join(output_dir, 'Log')
    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)
    mkdir_p(log_dir)

    logger = get_logger(output_dir)
    logger.info(cfg)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 80 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()
        ])
    dataset = TextDataset(cfg.DATA_DIR, 'train',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform,
                          mask=cfg.TRAIN.MASK_COND, audio_flag=args.audio_flag)

    print(dataset.n_words, dataset.embeddings_num)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #
    dataset_val = TextDataset(cfg.DATA_DIR, 'test',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform,
                              mask=cfg.TRAIN.MASK_COND, audio_flag=args.audio_flag)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, drop_last=True,
        shuffle=False, num_workers=int(cfg.WORKERS))
    writer = SummaryWriter(logdir=log_dir)
    # Train ##############################################################
    text_encoder, image_encoder, labels, start_epoch = build_models(dataset, batch_size, args.audio_flag)
    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)
    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    # At any point you can hit Ctrl + C to break out of training early.
    if args.eval:  # eval mode
        # eval last time
        logger.info('-' * 89)
        logger.info('eval at epoch: {}'.format(start_epoch))
        if len(dataloader_val) > 0:
            s_loss, w_loss = evaluate(dataloader_val, image_encoder,
                                      text_encoder, batch_size, writer, 0, dataset.ixtoword, labels, image_dir)
            logger.info('| end epoch {:3d} | valid loss '
                  's_loss: {:6.4f} w_loss: {:6.4f} |'
                  .format(cfg.TRAIN.MAX_EPOCH, s_loss, w_loss))
        logger.info('-' * 89)
        return
    else: # train mode
        pass

    try:
        lr = cfg.TRAIN.ENCODER_LR
        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            epoch_start_time = time.time()
            count = train(dataloader, image_encoder, text_encoder,
                          batch_size, labels, optimizer, epoch,
                          dataset.ixtoword, image_dir, writer, logger, args.update_interval)
            logger.info('-' * 89)
            if len(dataloader_val) > 0:
                s_loss, w_loss = evaluate(dataloader_val, image_encoder,
                                          text_encoder, batch_size, writer, count, dataset.ixtoword, labels, image_dir)
                logger.info('| end epoch {:3d} | valid loss '
                      '{:6.4f} {:6.4f} | lr {:.5f}|'
                      .format(epoch, s_loss, w_loss, lr))
            logger.info('-' * 89)
            if lr > cfg.TRAIN.ENCODER_LR/10.:
                lr *= 0.98

            if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
                epoch == cfg.TRAIN.MAX_EPOCH-1):
                torch.save(image_encoder.state_dict(),
                           '%s/image_encoder%d.pth' % (model_dir, epoch))
                torch.save(text_encoder.state_dict(),
                           '%s/text_encoder%d.pth' % (model_dir, epoch))
                logger.info('Save G/Ds models.')

    except KeyboardInterrupt:
        logger.info('-' * 89)
        logger.info('Exiting from training early')


if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    
    # print('Using config:')
    # pprint.pprint(cfg)
    main(args)

    
