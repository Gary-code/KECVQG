from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
from lmdbdict import lmdbdict
from lmdbdict.methods import DUMPS_FUNC, LOADS_FUNC
import os
import numpy as np
import numpy.random as npr
import random
from functools import partial

import torch
import torch.utils.data as data
from torch import nn
from PIL import Image
import multiprocessing
import six
import cv2
from torchvision import transforms
import time
import re


class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.

    in_memory: if in_memory is True, we save all the features in memory
               For individual np(y|z)s, we don't need to do that because the system will do this for us.
               Should be useful for lmdb or h5.
               (Copied this idea from vilbert)
    """

    def __init__(self, db_path, ext, in_memory=False):
        self.db_path = db_path
        self.ext = ext
        if self.ext == '.npy':
            self.loader = lambda x: np.load(six.BytesIO(x))
        else:
            def load_npz(x):
                x = np.load(six.BytesIO(x))
                return x['feat'] if 'feat' in x else x[
                    'z']  # normally it should be 'feat', but under cocotest_bu, the key is saved to be 'z' mistakenly.

            self.loader = load_npz
        if db_path.endswith('.lmdb'):
            self.db_type = 'lmdb'
            self.lmdb = lmdbdict(db_path, unsafe=True)
            self.lmdb._key_dumps = DUMPS_FUNC['ascii']
            self.lmdb._value_loads = LOADS_FUNC['identity']
        elif db_path.endswith('.pth'):  # Assume a key,value dictionary
            self.db_type = 'pth'
            self.feat_file = torch.load(db_path)
            self.loader = lambda x: x
            print('HybridLoader: ext is ignored')
        elif db_path.endswith('h5'):
            self.db_type = 'h5'
            self.loader = lambda x: np.array(x).astype('float32')
        else:
            self.db_type = 'dir'

        self.in_memory = in_memory
        if self.in_memory:
            self.features = {}

    def get(self, key):

        if self.in_memory and key in self.features:
            # We save f_input because we want to save the
            # compressed bytes to save memory
            f_input = self.features[key]
        elif self.db_type == 'lmdb':
            f_input = self.lmdb[key]
        elif self.db_type == 'pth':
            f_input = self.feat_file[key]
        elif self.db_type == 'h5':
            f_input = h5py.File(self.db_path, 'r')[key]
        else:
            f_input = open(os.path.join(self.db_path, key + self.ext), 'rb').read()

        if self.in_memory and key not in self.features:
            self.features[key] = f_input

        # load image
        feat = self.loader(f_input)

        return feat


class COCOh5Loader:
    def __init__(self, h5_path):
        print('\nCOCOh5Loader loading h5 file: ', h5_path)
        self.h5 = h5py.File(h5_path, 'r')

        self.features = self.h5['features']     # (123287, 2048, 36)
        self.boxes = self.h5['boxes']           # (123287, 4, 36)
        self.ids = self.h5['ids'][()]           # (123287,)
        # print('id', self.ids[0])
        self.objects_id = self.h5['objects_id'] # (123287, 36)
        self.heights = self.h5['heights']       # (123287,)
        self.widths = self.h5['widths']         # (123287,)

        self.id2index = {id: i for i, id in enumerate(self.ids)}

    def get(self, id):
        i = self.id2index[id]

        return (self.features[i].transpose(),
                self.boxes[i].transpose(),
                self.objects_id[i],
                self.heights[i],
                self.widths[i])


class AugmentedImageLoader:
    def __init__(self, opt):
        self.memory = {}
        self.opt = opt
    
    def get(self, id: int):
        return np.load(f'{self.opt.augmented_image_folder}/{id}.npy')

class KnowledgeLoader:
    def __init__(self, opt):
        self.dataset_name = opt.dataset_name
        self.opt = opt
        
        self.ques_id2ans = {}
        if self.dataset_name == 'OKVQA':
            ktrain = json.load(open(opt.okvqa_train_kb))
            kval = json.load(open(opt.okvqa_val_kb))
            ktrain.update(kval)
            self.OKVQA_knowledge = ktrain
            
            # preprocess answers
            atrain = json.load(open(opt.okvqa_train_annotations))['annotations']
            aval = json.load(open(opt.okvqa_val_annotations))['annotations']
            all_ans = atrain + aval
            for item in all_ans:
                ques_id = str(item['question_id'])
                ans_infos = item['answers']
                ans_candidates = [info['answer'].strip().lower() for info in ans_infos]
                ans_candidates = list(set(ans_candidates))
                self.ques_id2ans[ques_id] = ans_candidates

        elif self.dataset_name == 'VQA2.0':
            train_infos = json.load(open(opt.vqa20_train_annotations))
            val_infos = json.load(open(opt.vqa20_val_annotations))
            all_infos = train_infos + val_infos

            for item in all_infos:
                ques_id = str(item['question_id'])
                ans = item['answer'].strip().lower()
                self.ques_id2ans[ques_id] = ans
        else:
            assert False, 'Wrong dataset name.'

    

    def get_ans(self, ques_id):
        ans_candidates = self.ques_id2ans[str(ques_id)]

        # OKVQA dataset has multiple answers
        if self.dataset_name == 'OKVQA':
            idx = random.randint(0, len(ans_candidates) - 1)
            ans = ans_candidates[idx]
        else:
            ans = ans_candidates
        ans = re.compile('[^a-z0-9 ]*').sub('', ans)
        return ans

    
    def get(self, ques_id, index=None):
        if self.dataset_name == 'OKVQA':
            if(self.opt.okvqa_knowledge_source == 'ConceptNet'):
                # ConceptNet
                with open(f'{self.opt.okvqa_cn_kb_folder}/{ques_id}.txt', 'r') as f:
                    knowledges = f.readlines()
                    ans = self.get_ans(ques_id)
                    ans_know = [self.get_ans(ques_id) + ' ' + know for know in knowledges]
                    return ans_know
            else:
                # GPT-3
                knowledges = self.OKVQA_knowledge[str(ques_id)]
                ans_know = [self.get_ans(ques_id) + ' ' + know for know in knowledges]
                return ans_know
        
        elif self.dataset_name == 'VQA2.0':
            with open(f'{self.opt.vqa20_knowledges_folder}/{ques_id}.txt', 'r') as f:
                knowledges = f.readlines()
                ans = self.get_ans(ques_id)
                ans_know = [ans + ' ' + know for know in knowledges]
                return ans_know
        else:
            assert False, 'Wrong dataset name.'

class Dataset(data.Dataset):

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt

        self.use_knowledge = opt.use_know > 0
        if self.use_knowledge:
            self.knowledge_loader = KnowledgeLoader(self.opt)
            
        self.seq_per_img = opt.seq_per_img

        # feature related options
        self.use_fc = getattr(opt, 'use_fc', True)
        self.use_att = getattr(opt, 'use_att', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)
        # self.glove_embedding_dict = opt.glove_embedding_dict
        # load the json file which contains additional information about the dataset
        
        assert (opt.dataset_name in ['VQA2.0', 'OKVQA']), 'Wrong dataset name.'
        input_json = opt.input_json
        id2QA_json = opt.input_id2QA

        print('\nDataLoader loading json file: ', input_json)
        self.info = json.load(open(input_json))

        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('\nVocab size is ', self.vocab_size)

        print('\nDataLoader loading json file: ', id2QA_json)
        self.id2QA = json.load(open(id2QA_json)) # {question_id : [question, ans]}

        QA = self.id2QA[list(self.id2QA.keys())[0]]
        self.seq_length = len(QA[0])
        self.ans_length = len(QA[1])
        print(f'\nQuestions length = {self.seq_length} Answers length = {self.ans_length}')

        self.COCOh5Loader = COCOh5Loader(opt.coco_h5)
        self.AugImgLoader = AugmentedImageLoader(opt)
        
        # open the hdf5 file
        print('\nDataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir)

        self.data_in_memory = getattr(opt, 'data_in_memory', False)
        self.fc_loader = HybridLoader(self.opt.input_fc_dir, '.npy', in_memory=self.data_in_memory)
        self.att_loader = HybridLoader(self.opt.input_att_dir, '.npz', in_memory=self.data_in_memory)
        self.box_loader = HybridLoader(self.opt.input_box_dir, '.npy', in_memory=self.data_in_memory)
        # self.iod_loader = HybridLoader(self.opt.input_att_dir_iod, '.npy', in_memory=self.data_in_memory)

        self.num_images = len(self.info['images'])  # self.label_start_ix.shape[0]
        print('\nRead %d QA pairs' % (self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]

            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['test'].append(ix)
            else:
                raise RuntimeError('split NOT in image info.')

        # self.split_ix['test'] = self.split_ix['test'][0:5000]
        print('\nAssigned %d QA pairs to split TRAIN' % len(self.split_ix['train']))
        print('\nAssigned %d QA pairs to split VAL' % len(self.split_ix['test']))


    def collate_func(self, batch, split):
        seq_per_img = self.seq_per_img

        knowledge_batch = []
        aug_feat_batch = []
        fc_batch = []
        att_batch = []
        # iod_batch=[]
        label_batch = []
        answer_batch = []
        objects_id_batch = []
        boxes_batch = []

        wrapped = False

        infos = []
        gts = []

        for i, sample in enumerate(batch):
            # fetch image
            knowledge, aug_feat, tmp_fc, tmp_att, tmp_seq, ix, it_pos_now, tmp_wrapped, tmp_ans, objects_id, boxes = sample

            if tmp_wrapped:
                wrapped = True
            if self.use_knowledge:
                for k in knowledge:
                    knowledge_batch.append(k)
            aug_feat_batch.append(aug_feat)
            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)
            # iod_batch.append(tmp_iod)
            objects_id_batch.append(objects_id)
            boxes_batch.append(boxes)

            # tmp_label = np.array(tmp_seq, dtype='int')
            tmp_answer = np.array(tmp_ans, dtype='int')

            tmp_label = np.zeros([seq_per_img, self.seq_length + 2], dtype='int')
            tmp_label[:, 1: self.seq_length + 1] = tmp_seq
            # # print('tmp_label.shape', tmp_label.shape)
            
            # tmp_answer = np.zeros([seq_per_img, self.ans_length], dtype='int')
            # tmp_answer[:, :] = tmp_ans

            label_batch.append(tmp_label)
            answer_batch.append(tmp_answer)

            # Used for reward evaluation
            gts.append(self.id2QA[str(self.info['images'][ix]['qid'])][0])

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix].get('file_path', '')
            infos.append(info_dict)
        
        data = {}
        data['fc_feats'] = np.stack(fc_batch)

        max_att_len = max([_.shape[0] for _ in att_batch])

        data['knowledges'] = knowledge_batch
        data['aug_feats'] = np.array(aug_feat_batch, dtype='float32')

        data['att_feats'] = np.zeros([len(att_batch), max_att_len, att_batch[0].shape[1]], dtype='float32')
        for i in range(len(att_batch)):
            data['att_feats'][i, :att_batch[i].shape[0]] = att_batch[i]

        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i, :att_batch[i].shape[0]] = 1

        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None
            data['iod_masks'] = None

        data['objects_id'] = np.array(objects_id_batch, dtype='int32')
        data['boxes'] = np.array(boxes_batch, dtype='float32')

        data['labels'] = np.vstack(label_batch)
        # print('data[labels].shape', data['labels'].shape)
        data['answers'] = np.vstack(answer_batch)

        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
        mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 2], dtype='float32')
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        
        data['masks'] = mask_batch
        data['labels'] = data['labels'].reshape(len(batch), seq_per_img, -1)
        data['answers'] = data['answers'].reshape(len(batch), seq_per_img, -1)
        data['masks'] = data['masks'].reshape(len(batch), seq_per_img, -1)

        data['gts'] = gts  # all ground truth of each images
        data['bounds'] = {'it_pos_now': it_pos_now,  # the it_pos_now of the last sample
                          'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor

        return data

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix, it_pos_now, wrapped = index  # self.split_ix[index]

        img_id = int(self.info['images'][ix]['id'])
        features, boxes, objects_id, h, w = self.COCOh5Loader.get(img_id)
        aug_features = self.AugImgLoader.get(img_id)

        if self.use_att:
            att_feat = features 
            
            # Reshape to K x C
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])

            if self.norm_att_feat:
                att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
                aug_features = aug_features / np.linalg.norm(aug_features, 2, 1, keepdims=True)

            if self.use_box: # default is False
                box_feat = boxes
                # devided by image width and height
                x1, y1, x2, y2 = np.hsplit(box_feat, 4)

                # h, w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
                box_feat = np.hstack(
                    (x1 / w, y1 / h, x2 / w, y2 / h, (x2 - x1) * (y2 - y1) / (w * h)))  # question? x2-x1+1??

                if self.norm_box_feat:
                    box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
                                
                # sort the features by the size of boxes
                index = sorted(range(len(box_feat)), key = lambda x:box_feat[x][-1], reverse=True)
                print('\natt_feat shape before: ', att_feat.shape)
                att_feat = att_feat[index]
                print('\natt_feat shape after: ', att_feat.shape)
                box_feat = box_feat[index]
                att_feat = np.hstack([att_feat, box_feat])
                objects_id = objects_id[index]
                boxes = boxes[index]
        else:
            att_feat = np.zeros((0, 0), dtype='float32')
            aug_features = np.zeros((0, 0), dtype='float32')

        if self.use_fc: # KECVQG is False
            try:
                fc_feat = self.fc_loader.get(str(self.info['images'][ix]['id']))
            except:
                # Use average of attention when there is no fc provided (For bottomup feature)
                fc_feat = att_feat.mean(0)
        else:
            fc_feat = np.zeros((0), dtype='float32')

        qid = str(self.info['images'][ix]['qid'])
        QA = self.id2QA[qid]

        seq = QA[0]
        ans = QA[1]
        
        know = None
        if self.use_knowledge:
            know = self.knowledge_loader.get(qid) # list of 5 strings(ans+knowledge)

        return (know, aug_features, fc_feat,
                att_feat, seq, ix, it_pos_now, wrapped, ans, objects_id, boxes)

    def __len__(self):
        return len(self.info['images'])


class DataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.dataset = Dataset(opt)
        
        # Initialize loaders and iters
        self.loaders, self.iters = {}, {}
        for split in ['train', 'val', 'test']:
            if split == 'train':
                sampler = MySampler(self.dataset.split_ix[split], shuffle=True, wrap=True)
            else:
                sampler = MySampler(self.dataset.split_ix[split], shuffle=False, wrap=False)

            self.loaders[split] = data.DataLoader(dataset=self.dataset,
                                                  batch_size=self.batch_size,
                                                  sampler=sampler,
                                                  pin_memory=True,
                                                  num_workers=4,  # 4 is usually enough
                                                  collate_fn=partial(self.dataset.collate_func, split=split),
                                                  drop_last=False)
            self.iters[split] = iter(self.loaders[split])


    def get_batch(self, split):
        try:
            data = next(self.iters[split])
        except StopIteration:
            self.iters[split] = iter(self.loaders[split])
            data = next(self.iters[split])
        return data

    def reset_iterator(self, split):
        self.loaders[split].sampler._reset_iter()
        self.iters[split] = iter(self.loaders[split])

    def get_vocab_size(self):
        return self.dataset.get_vocab_size()

    @property
    def vocab_size(self):
        return self.get_vocab_size()

    def get_vocab(self):
        return self.dataset.get_vocab()

    def get_seq_length(self):
        return self.dataset.get_seq_length()

    @property
    def seq_length(self):
        return self.get_seq_length()

    def state_dict(self):
        def get_prefetch_num(split):
            if self.loaders[split].num_workers > 0:
                return (self.iters[split]._send_idx - self.iters[split]._rcvd_idx) * self.batch_size
            else:
                return 0

        return {split: loader.sampler.state_dict(get_prefetch_num(split)) \
                for split, loader in self.loaders.items()}

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        for split in self.loaders.keys():
            self.loaders[split].sampler.load_state_dict(state_dict[split])


class MySampler(data.sampler.Sampler):
    def __init__(self, index_list, shuffle, wrap):
        self.index_list = index_list
        self.shuffle = shuffle
        self.wrap = wrap
        # if wrap, there will be not stop iteration called
        # wrap True used during training, and wrap False used during test.
        self._reset_iter()

    def __iter__(self):
        return self

    def __next__(self):
        wrapped = False
        if self.iter_counter == len(self._index_list):
            self._reset_iter()
            if self.wrap:
                wrapped = True
            else:
                raise StopIteration()
        if len(self._index_list) == 0:  # overflow when 0 samples
            return None
        elem = (self._index_list[self.iter_counter], self.iter_counter + 1, wrapped)
        self.iter_counter += 1
        return elem

    def next(self):
        return self.__next__()

    def _reset_iter(self):
        if self.shuffle:
            rand_perm = npr.permutation(len(self.index_list))
            self._index_list = [self.index_list[_] for _ in rand_perm]
        else:
            self._index_list = self.index_list

        self.iter_counter = 0

    def __len__(self):
        return len(self.index_list)

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        self._index_list = state_dict['index_list']
        self.iter_counter = state_dict['iter_counter']

    def state_dict(self, prefetched_num=None):
        prefetched_num = prefetched_num or 0
        return {
            'index_list': self._index_list,
            'iter_counter': self.iter_counter - prefetched_num
        }
