from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from vqg.modules.loss_wrapper import LossWrapper
from vqg.utils.rewards import init_scorer
import vqg.utils.misc as utils
import vqg.utils.eval_utils as eval_utils
import skimage.io
from vqg.data.dataloader import *
import vqg.models as models
import vqg.utils.opts as opts
from collections import defaultdict
import traceback
from six.moves import cPickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import pytz

from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

import os
os.environ['PYTHONPATH'] = '.'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


def train(opt):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    ################################
    # Build dataloader
    ################################
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    ##########################
    # Initialize infos
    ##########################
    infos = {
        'iter': 0,
        'epoch': 0,
        'loader_state_dict': None,
        'vocab': loader.get_vocab(),
    }

    infos['opt'] = opt

    #########################
    # Build logger
    #########################
    # naive dict logger
    histories = defaultdict(dict)
    # if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')):
    #     with open(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl'), 'rb') as f:
    #         histories.update(utils.pickle_load(f))

    # tensorboard logger
    tb_summary_writer = SummaryWriter(opt.checkpoint_path)

    ##########################
    # Build model
    ##########################
    opt.vocab = loader.get_vocab()
    model = models.setup(opt, tokenizer).cuda()
    del opt.vocab
    # Load pretrained weights:
    # if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'model.pth')):
    #     model.load_state_dict(torch.load(
    #         os.path.join(opt.start_from, 'model.pth')))

    # Wrap generation model with loss function(used for training)
    # This allows loss function computed separately on each machine
    lw_model = LossWrapper(model, opt)
    # Wrap with dataparallel
    dp_model = torch.nn.DataParallel(model, device_ids=[0])
    # dp_model = lw_model
    dp_model.vocab = getattr(model, 'vocab', None)  # nasty
    dp_lw_model = torch.nn.DataParallel(lw_model, device_ids=[0])
    # dp_lw_model = dp_model

    ##########################
    #  Build optimizer
    ##########################
    if opt.noamopt:
        assert opt.vqg_model in ['transformer', 'bert', 'm2transformer', 'KECVQG'], 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(
            model, optim_func=opt.optim, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)

    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer,
                                            factor=opt.reduce_on_plateau_factor,
                                            patience=opt.reduce_on_plateau_patience)
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)

    # Load the optimizer
    # if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, "optimizer.pth")):
    #     optimizer.load_state_dict(torch.load(
    #         os.path.join(opt.start_from, 'optimizer.pth')))

    #########################
    # Get ready to start
    #########################
    iteration = infos['iter']
    epoch = infos['epoch']
    # For back compatibility
    if 'iterators' in infos:
        infos['loader_state_dict'] = {
            split: {'index_list': infos['split_ix'][split], 'iter_counter': infos['iterators'][split]} for split in
            ['train', 'val', 'test']}
    loader.load_state_dict(infos['loader_state_dict'])

    best_val_score = None
    # if opt.load_best_score == 1:
    #     best_val_score = infos.get('best_val_score', None)
    if opt.noamopt:
        optimizer._step = iteration
    # flag indicating finish of an epoch
    # Always set to True at the beginning to initialize the lr or etc.
    epoch_done = True
    # Assure in training mode
    dp_lw_model.train()
    # Start training

    start_time = datetime.now().astimezone(pytz.timezone(
        'Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')

    try:
        while True:
            # Stop if reaching max epochs
            if epoch >= opt.max_epochs and opt.max_epochs != -1:
                break

            if epoch_done:
                if not opt.noamopt and not opt.reduce_on_plateau:
                    # Assign the learning rate
                    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                        frac = (
                            epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                        decay_factor = opt.learning_rate_decay_rate ** frac
                        opt.current_lr = opt.learning_rate * decay_factor
                    else:
                        opt.current_lr = opt.learning_rate
                    # set the decayed rate
                    utils.set_lr(optimizer, opt.current_lr)
                # Assign the scheduled sampling prob
                if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (
                        epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(
                        opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob

                sc_flag = False
                epoch_done = False

            if opt.use_warmup and (iteration < opt.noamopt_warmup):
                opt.current_lr = opt.learning_rate * \
                    (iteration + 1) / opt.noamopt_warmup
                utils.set_lr(optimizer, opt.current_lr)
            # Load data from train split (0)
            start = time.time()
            data = loader.get_batch('train')
            read_data_time = time.time() - start

            torch.cuda.synchronize()
            start = time.time()

            tmp = [data['aug_feats'], data['fc_feats'], data['att_feats'], data['objects_id'], data['boxes'], data['labels'],
                   data['answers'], data['masks'], data['att_masks'], data['iod_masks']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]

            aug_feats, fc_feats, att_feats, objects_id, boxes, labels, answers, masks, att_masks, iod_masks = tmp
            # print('labels.shape', labels.shape)

            optimizer.zero_grad()

            model_out = dp_lw_model(data['knowledges'], aug_feats, att_feats, objects_id, boxes, labels, answers, masks, att_masks, iod_masks, data['gts'],
                                    torch.arange(0, len(data['gts'])), sc_flag)

            # accumulate_iter = accumulate_iter + 1
            loss = model_out['loss']

            if 'loss_att' in model_out:
                loss += model_out['loss_att']
            if 'loss_eff' in model_out:
                loss += model_out['loss_eff'] + model_out['loss_sel']

            loss.sum().backward()

            # if accumulate_iter % opt.accumulate_number == 0:
            if opt.grad_clip_value != 0:
                getattr(torch.nn.utils, 'clip_grad_%s_' % (opt.grad_clip_mode))(
                    model.parameters(), opt.grad_clip_value)
            optimizer.step()
            train_loss = model_out['loss'].sum().item()

            loss_att = model_out['loss_att'].sum().item(
            ) if 'loss_att' in model_out else -1.0

            loss_eff = model_out['loss_eff'].sum(
            ).item() if 'loss_eff' in model_out else -1.0

            loss_sel = model_out['loss_sel'].sum(
            ).item() if 'loss_sel' in model_out else -1.0

            torch.cuda.synchronize()
            end = time.time()

            print('\nDataset {}, Tag {}, BLEU4 {:.3E}, GPU {}, BS {}, Eval/iter {}'.format(opt.dataset_name, opt.tag, 0 if best_val_score is None else best_val_score,
                                                                                           opt.GPU, opt.batch_size, opt.SCE))

            print("Read data {:.2f}, Iter {}, Epoch {}, L_KAI {:.2f}, L_Att {:.2f}, L_Eff {:.2f}, L_Sel {:.2f}, Time/batch {:.2f}"
                  .format(read_data_time, iteration, epoch, train_loss, loss_att, loss_eff, loss_sel,
                          end - start, 0 if best_val_score is None else best_val_score,
                          opt.dataset_name, opt.GPU, opt.batch_size))

            # Update the iteration and epoch
            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1
                epoch_done = True

            # make evaluation on validation set, and save model
            if ((iteration) % opt.SCE == 0):
                # eval model
                eval_kwargs = {'split': opt.split, 'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))
                val_loss, predictions, lang_stats, bleu4_score = eval_utils.eval_split(
                    opt, start_time, iteration, dp_model, lw_model.crit, loader, eval_kwargs, best_val_score, model.VD,
                    model.causal_factorization, tokenizer)

                if best_val_score is None or bleu4_score > best_val_score:
                    best_val_score = bleu4_score

                if opt.reduce_on_plateau:
                    if 'CIDEr' in lang_stats:
                        optimizer.scheduler_step(-lang_stats['CIDEr'])
                    else:
                        optimizer.scheduler_step(val_loss)

            # Write the training loss summary
            if (iteration % opt.losses_log_every == 0):
                tb_summary_writer.add_scalar(
                    'train_loss', train_loss, iteration)
                if opt.noamopt:
                    opt.current_lr = optimizer.rate()
                elif opt.reduce_on_plateau:
                    opt.current_lr = optimizer.current_lr
                tb_summary_writer.add_scalar(
                    'learning_rate', opt.current_lr, iteration)
                tb_summary_writer.add_scalar(
                    'scheduled_sampling_prob', model.ss_prob, iteration)
                if sc_flag:
                    tb_summary_writer.add_scalar(
                        'avg_reward', model_out['reward'].mean(), iteration)

                histories['loss_history'][iteration] = train_loss if not sc_flag else model_out['reward'].mean()
                histories['lr_history'][iteration] = opt.current_lr
                histories['ss_prob_history'][iteration] = model.ss_prob

            # update infos
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['loader_state_dict'] = loader.state_dict()

    except (RuntimeError, KeyboardInterrupt):
        # print('Save ckpt on exception ...')
        # utils.save_checkpoint(opt, model, infos, optimizer)
        # print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)


opt = opts.parse_opt()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.GPU
if opt.dataset_name == 'OKVQA':
    opt.SCE = ((9009 // 2) // opt.batch_size) + 1
elif opt.dataset_name == 'VQA2.0':
    opt.SCE = ((250001 // 2) // opt.batch_size) + 1

# os.environ['export CUDA_LAUNCH_BLOCKING'] = '1'
print(f'\nUsing GPU device: {opt.GPU} Batch Size = {opt.batch_size}')
train(opt)
