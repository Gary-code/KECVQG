from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import re
from pycocoevalcap.bleu.bleu import Bleu
from datetime import datetime
from . import misc as utils
import pytz

try:
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
except:
    print('Warning: not available')

bad_endings = ['a', 'an', 'the', 'in', 'for', 'at', 'of', 'with', 'before', 'after', 'on', 'upon', 'near', 'to', 'is',
               'are', 'am']
bad_endings += ['the']


def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0


def getCOCO(dataset):
    if 'coco' in dataset:
        annFile = '/home/liu/self-critical.pytorch-master/coco-caption/annotations/captions_val2014.json'
    elif 'flickr30k' in dataset or 'f30k' in dataset:
        annFile = 'data/f30k_captions4eval.json'
    return COCO(annFile)


def language_eval(dataset, preds, preds_n, eval_kwargs, split):

    with open('./myscripts/preds.json', 'w') as outfile:
        json.dump({'dataset': dataset, 'preds': preds, 'preds_n': preds_n, 'eval_kwargs': eval_kwargs, 'split': split}, outfile)

    model_id = eval_kwargs['id']
    eval_oracle = eval_kwargs.get('eval_oracle', 0)

    # create output dictionary
    out = {}

    if len(preds_n) > 0:
        # vocab size and novel sentences
        if 'coco' in dataset:
            dataset_file = '/home/liu/self-critical.pytorch-master/data/cocotalk.json'
        elif 'flickr30k' in dataset or 'f30k' in dataset:
            dataset_file = 'data/dataset_flickr30k.json'
        training_sentences = set([' '.join(__['tokens']) for _ in json.load(open(dataset_file))['images'] if
                                  not _['split'] in ['val', 'test'] for __ in _['sentences']])
        generated_sentences = set([_['caption'] for _ in preds_n])
        novels = generated_sentences - training_sentences
        out['novel_sentences'] = float(len(novels)) / len(preds_n)
        tmp = [_.split() for _ in generated_sentences]
        words = []
        for _ in tmp:
            words += _
        out['vocab_size'] = len(set(words))

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    cache_path = os.path.join('eval_results/', '.cache_' + model_id + '_' + split + '.json')

    coco = getCOCO(dataset)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set
    preds_filt = [p for p in preds if p['image_id'] in valids]
    mean_perplexity = sum([_['perplexity'] for _ in preds_filt]) / len(preds_filt)
    mean_entropy = sum([_['entropy'] for _ in preds_filt]) / len(preds_filt)
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        out[metric] = score
    # Add mean perplexity
    out['perplexity'] = mean_perplexity
    out['entropy'] = mean_entropy

    imgToEval = cocoEval.imgToEval
    for k in list(imgToEval.values())[0]['SPICE'].keys():
        if k != 'All':
            out['SPICE_' + k] = np.array([v['SPICE'][k]['f'] for v in imgToEval.values()])
            out['SPICE_' + k] = (out['SPICE_' + k][out['SPICE_' + k] == out['SPICE_' + k]]).mean()
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    if len(preds_n) > 0:
        from . import eval_multi
        cache_path_n = os.path.join('eval_results/', '.cache_' + model_id + '_' + split + '_n.json')
        allspice = eval_multi.eval_allspice(dataset, preds_n, model_id, split)
        out.update(allspice['overall'])
        div_stats = eval_multi.eval_div_stats(dataset, preds_n, model_id, split)
        out.update(div_stats['overall'])
        if eval_oracle:
            oracle = eval_multi.eval_oracle(dataset, preds_n, model_id, split)
            out.update(oracle['overall'])
        else:
            oracle = None
        self_cider = eval_multi.eval_self_cider(dataset, preds_n, model_id, split)
        out.update(self_cider['overall'])
        with open(cache_path_n, 'w') as outfile:
            json.dump({'allspice': allspice, 'div_stats': div_stats, 'oracle': oracle, 'self_cider': self_cider},
                      outfile)

    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def bleu(gts, res):
    scorer = Bleu(n=4)
    # scorer += (hypo[0], ref1)   # hypo[0] = 'word1 word2 word3 ...'
    #                                 # ref = ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    score, scores = scorer.compute_score(gts, res)

    # print('belu = %s' % score)
    return score[-1]

def calculateBLEU4(predictions, opt):
    questionTrain = json.load(open(opt.vqa20_train_annotations))
    questionVal = json.load(open(opt.vqa20_val_annotations))
    qlist = questionTrain + questionVal

    id2question = {}
    for item in qlist:
        id = int(item['image_id'])
        if id not in id2question:
            id2question[id] = []
        q = str(item['question'])
        q = re.compile('[^a-z0-9 ]*').sub('', q.lower())
        q = ' '.join(q.split())
        id2question[id].append(q)

    questions = []
    max_gts = []
    imgids = []
    for i, item in enumerate(predictions):
        id = int(item['image_id'])
        if id not in id2question:
            continue
        imgids.append(id)
        q_pred = str(item['caption']).lower()
        questions.append(q_pred)
        q_gts = id2question[id]

        scores = [bleu({i: [q_gt]}, {i: [q_pred]}) for q_gt in q_gts]
        print(f'scores = {scores}')
        max_idx = scores.index(max(scores))
        max_gts.append(q_gts[max_idx])

    q_preds = {i: [s] for i, s in enumerate(questions)}
    q_gts = {i: [s] for i, s in enumerate(max_gts)}

    final_score = bleu(q_gts, q_preds)
    print(f'Num questions = {len(questions)} MAX BLEU4 = {final_score}')  # results, return bleu4 value
    return final_score, questions, max_gts, imgids


def eval_split(opt, start_time, iter, model, crit, loader, eval_kwargs, best_val_score, VD, causal_factorization, tokenizer):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(
        remove_bad_endings)  # Use this nasty way to make other code clean since it's a global configuration
    device = eval_kwargs.get('device', 'cuda')

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    n_predictions = []  # when sample_n > 1

    dir_path = f'./results/{opt.dataset_name}/{opt.tag}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    log = open(f'{dir_path}/log.txt', 'a+')

    while True:
        data = loader.get_batch(split)
        n = n + len(data['infos'])

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['answers'], data['masks'], data['att_masks'], data['iod_masks'], data['boxes'], data['objects_id']]
        tmp = [_.to(device) if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, labels, answers, masks, att_masks, iod_masks, boxes, objects_id = tmp

        if opt.enable_VD:
            iod_feats = VD(att_feats, boxes, objects_id)
        else:
            iod_feats = att_feats.clone()

        if opt.enable_AFM:
            aug_feats = causal_factorization(att_feats)
            iod_feats = torch.cat([iod_feats, aug_feats], dim=2)

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            tmp_eval_kwargs = eval_kwargs.copy()
            tmp_eval_kwargs.update({'sample_n': 1})
            
            seq, seq_logprobs = model(data['knowledges'], fc_feats, att_feats, iod_feats, answers, att_masks, iod_masks, opt=tmp_eval_kwargs, mode='sample')
            seq = seq.data # tensor of shape(bs, seq_len)
            entropy = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / (
                        (seq > 0).to(seq_logprobs).sum(1) + 1)
            perplexity = - seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(2).sum(1) / (
                        (seq > 0).to(seq_logprobs).sum(1) + 1)

        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(fc_feats.shape[0]):
                print('\n'.join(
                    [utils.decode_sequence(model.vocab, _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)

        # sents = utils.decode_sequence(model.vocab, seq) # list of strings
        # gts = utils.decode_sequence(model.vocab, labels.view(-1, labels.shape[-1])) # list of strings
        
        with torch.no_grad():
            sents = tokenizer.batch_decode(seq, skip_special_tokens=True)
            gts = tokenizer.batch_decode(labels.view(-1, labels.shape[-1]), skip_special_tokens=True)
        
        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'gt': gts[k], 'perplexity': perplexity[k].item(),
                     'entropy': entropy[k].item()}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)

            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'],
                                            data['infos'][k]['file_path']) + '" vis/imgs/img' + str(
                    len(predictions)) + '.jpg'  # bit gross
                print(cmd)
                os.system(cmd)

            # print every result
            # if verbose:
            #     print('image %s: %s' % (entry['image_id'], entry['caption']))

        if sample_n > 1:
            assert False, 'why sample_n > 1?'
            eval_split_n(model, n_predictions, [fc_feats, att_feats, att_masks, data], eval_kwargs)

        # ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        else:
            num_images = ix1
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            sys.stdout.write('\revaluating validation preformance... %d/%d (%f)' % (n, ix1, loss))

        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    bleu4_score = None
    if len(n_predictions) > 0 and 'perplexity' in n_predictions[0]:
        n_predictions = sorted(n_predictions, key=lambda x: x['perplexity'])
    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    torch.save((predictions, n_predictions),
               os.path.join('eval_results/', '.saved_pred_' + eval_kwargs['id'] + '_' + split + '.pth'))

    lang_stats = None #直接不管它了

    stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w') # 禁用控制台输出
    q_preds = []
    q_gts = []
    imgids = []
    if opt.dataset_name != 'VQA2.0':
        for item in predictions:
            q_preds.append(item['caption'])
            q_gts.append(item['gt'])
            imgids.append(item['image_id'])
        
        q_preds2 = {i: [s] for i, s in enumerate(q_preds)}
        q_gts2 = {i: [s] for i, s in enumerate(q_gts)}
        bleu4_score = bleu(q_gts2, q_preds2)
    else:
        bleu4_score, q_preds, q_gts, imgids = calculateBLEU4(predictions, opt)
    sys.stdout = stdout # 恢复控制台输出

    log.write(f'{start_time}, iter={iter}, bleu4_score={bleu4_score:.5E}, best_val_score={0 if best_val_score is None else best_val_score:.5E}\n')

    #存储最优结果
    if best_val_score is None or bleu4_score > best_val_score:
        log.write(f'bleu4_score > best_val_score.\n')

        preds_output = open(f'{dir_path}/q_preds.txt', 'w', encoding='utf-8')
        gts_output = open(f'{dir_path}/q_gts.txt', 'w', encoding='utf-8')
        preds_with_imgid = open(f'{dir_path}/q_preds_imgid.txt', 'w', encoding='utf-8')
        
        for i in range(len(q_preds)):
            preds_with_imgid.write(str(imgids[i]))
            preds_with_imgid.write(', ')
            preds_with_imgid.write(q_preds[i])
            preds_with_imgid.write(', ')
            preds_with_imgid.write(q_gts[i])
            preds_with_imgid.write('\n')

            preds_output.write(q_preds[i])
            preds_output.write('\n')

            gts_output.write(q_gts[i])
            gts_output.write('\n')

        preds_output.close()
        gts_output.close()
        preds_with_imgid.close()

    # Switch back to training mode
    log.write(f'val_loss={loss_sum / loss_evals}\n\n')
    log.close()
    
    model.train()
    return loss_sum / loss_evals, predictions, lang_stats, bleu4_score


# Only run when sample_n > 0
def eval_split_n(model, n_predictions, input_data, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    sample_n_method = eval_kwargs.get('sample_n_method', 'sample')

    fc_feats, att_feats, att_masks, data = input_data

    tmp_eval_kwargs = eval_kwargs.copy()
    if sample_n_method == 'bs':
        # case 1 sample_n == beam size
        tmp_eval_kwargs.update({'sample_n': 1, 'beam_size': sample_n, 'group_size': 1})  # randomness from softmax
        with torch.no_grad():
            model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        for k in range(fc_feats.shape[0]):
            _sents = utils.decode_sequence(model.vocab,
                                           torch.stack([model.done_beams[k][_]['seq'] for _ in range(sample_n)]))
            for sent in _sents:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                n_predictions.append(entry)
    # case 2 sample / gumbel / topk sampling/ nucleus sampling
    elif sample_n_method == 'sample' or \
            sample_n_method == 'gumbel' or \
            sample_n_method.startswith('top'):
        tmp_eval_kwargs.update(
            {'sample_n': sample_n, 'sample_method': sample_n_method, 'beam_size': 1})  # randomness from sample
        with torch.no_grad():
            _seq, _sampleLogprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        _sents = utils.decode_sequence(model.vocab, _seq)
        _perplexity = - _sampleLogprobs.gather(2, _seq.unsqueeze(2)).squeeze(2).sum(1) / (
                    (_seq > 0).to(_sampleLogprobs).sum(1) + 1)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent,
                     'perplexity': _perplexity[k].item()}
            n_predictions.append(entry)
    elif sample_n_method == 'dbs':
        # Use diverse beam search
        tmp_eval_kwargs.update({'beam_size': sample_n * beam_size, 'group_size': sample_n})  # randomness from softmax
        with torch.no_grad():
            model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        for k in range(loader.batch_size):
            _sents = utils.decode_sequence(model.vocab, torch.stack(
                [model.done_beams[k][_]['seq'] for _ in range(0, sample_n * beam_size, beam_size)]))
            for sent in _sents:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                n_predictions.append(entry)
    else:
        tmp_eval_kwargs.update(
            {'sample_method': sample_n_method[1:], 'group_size': sample_n, 'beam_size': 1})  # randomness from softmax
        with torch.no_grad():
            _seq, _sampleLogprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        _sents = utils.decode_sequence(model.vocab, _seq)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent}
            n_predictions.append(entry)
    if verbose:
        for entry in sorted(n_predictions[-fc_feats.shape[0] * sample_n:], key=lambda x: x['image_id']):
            print('image %s: %s' % (entry['image_id'], entry['caption']))