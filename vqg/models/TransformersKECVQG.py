from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils

import copy
import math
import numpy as np
import random

from .VQGModel import VQGModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel

from transformers import BertModel, BertConfig, VisualBertConfig
from .VisualDeconfounder import VisualDeconfounder
from vqg.data.dataloader import *

from transformers import VisualBertModel


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    # att_feats, iod_feats, seq, answers, att_masks, iod_masks, seq_mask
    def forward(self, knowledges, src, iod_feats, tgt, answers, src_mask, iod_masks, tgt_mask):
        "Take in and process masked src and target sequences."
        memory_know, mask_know, memory, mask, match_pred = self.encode(
            knowledges, src, iod_feats, src_mask, iod_masks, answers)
        # print('memory_know', memory_know.shape, memory.shape)
        return self.decode(memory_know, mask_know, tgt, tgt_mask), self.decode(memory, mask, tgt, tgt_mask), match_pred

    def encode(self, knowledges, src, iod_feats, src_mask, iod_masks, answers):
        # return src
        return self.encoder(knowledges, self.src_embed(src), iod_feats, src_mask, iod_masks, answers)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ObjectAttention(nn.Module):
    def __init__(self, rnn_size, att_hid_size):
        super(ObjectAttention, self).__init__()
        self.rnn_size = rnn_size
        self.att_hid_size = att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size).cuda()
        self.alpha_net = nn.Linear(self.att_hid_size, 1).cuda()

    def forward(self, h, att_feats, att_masks=None):
        """
        :param h: word embedding [bs, rnn_size]
        :param att_feats:  object features [bs, max_obj_len, att_hid_size]
        :param att_masks:  object mask [bs, max_obj_len]
        :return:
        """
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = att_feats
        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(
            att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = torch.tanh(dot)  # batch * att_size * att_hid_size
        # (batch * att_size) * att_hid_size
        dot = dot.view(-1, self.att_hid_size)
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot, dim=1)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        # batch * att_size * att_feat_size
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))

        return att_feats_

class Selector(nn.Module):
    def __init__(self, topk, sim_func) -> None:
        """
        Args:
            topk (int): return topk most relevant knowledge
            sim_func (nn.Module): similarity function, default cosine similarity
        """
        super().__init__()
        self.topk = topk
        self.sim_func = sim_func

    @torch.no_grad()
    def forward(self, query, knowledge_embed, knowledge_full):
        """ Retrieve topk knoweldge in non-batch manner for resource reason. """
        topk_knowledge, topk_embed = [], []
        kb_len = knowledge_embed.shape[0]
        for query_single in query:
            # use other GPU to store the knowledge embedding if you have no enough GPU memory
            sims_1 = self.sim_func(query_single.unsqueeze(dim=0), knowledge_embed[:kb_len//2, :])
            sims_2 = self.sim_func(query_single.unsqueeze(dim=0).to('cuda:2'), knowledge_embed[kb_len//2:, :].to('cuda:2'))
            sims = torch.cat((sims_1.to(query_single.device), sims_2.to(query_single.device)), dim=0)
            indices = sims.topk(self.topk, dim=-1)[1]
            topk_embed.append(knowledge_embed[indices])
            topk_knowledge.append([knowledge_full[idx] for idx in indices])
        return topk_knowledge, torch.stack(topk_embed, dim=0)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N, opt, tokenizer):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

        self.do_ans_intervention = opt.ans_inte > 0
        if self.do_ans_intervention:
            # answer confounder
            self.ans_dict = torch.tensor(np.load(
                './data/ans_dict.npy'), dtype=torch.float)
            self.ans_prob = torch.tensor(np.load(
                './data/ans_probs.npy'), dtype=torch.float)

        cfg = BertConfig.from_pretrained("bert-base-uncased")
        cfg.visual_embedding_dim = 512
        cfg.hidden_size = 512
        cfg.num_attention_heads = 8
        self.bert_model = BertModel(cfg)

        self.ans_attention = opt.ans_atte > 0
        if self.ans_attention:
            self.object_attention = ObjectAttention(512, 512)

        self.use_knowledge = opt.use_know > 0
        if self.use_knowledge:
            self.tokenizer = tokenizer  # AutoTokenizer
            self.sim_func = nn.CosineSimilarity(dim=-1)
            self.selector = Selector(topk=3, sim_func=self.sim_func)
            config = VisualBertConfig.from_pretrained(
                "uclanlp/visualbert-vqa-coco-pre")
            # print(config)
            config.visual_embedding_dim = 512
            config.hidden_size = 512
            config.num_attention_heads = 8
            self.VLM = VisualBertModel(config)
            self.knowledge_loader = KnowledgeLoader(opt)
            knowledge_full_path = './data/kb-data/knowledge.json'
            with open(knowledge_full_path, 'r') as fd:
                self.knowledge_full = json.load(fd)
            self.knowledge_embed = torch.load('./data/kb-data/knowledge_embed.pt').cuda()

    # src, iod_feats, src_mask, iod_masks, answers
    def forward(self, knowledges, x, y, mask, iod_mask, answers):
        # print('mask', mask.shape) # (bs, 1, 36)
        "Pass the input (and mask) through each layer in turn."
        if answers.ndim == 3:  # (bs, 1, ans_len)
            answers = answers.squeeze(1)

        x_copy = x.clone()
        mask_copy = mask.clone()

        # Answer Intervention
        if self.do_ans_intervention:
            x = self.ans_intervention(x)
            x_copy = self.ans_intervention(x_copy)

        # Answer & x(obj) Attention
        output = self.bert_model(answers)
        ans = output.pooler_output  # (bs, 768)
        if self.ans_attention:
            x = self.object_attention(ans, x)
            x_copy = self.object_attention(ans, x_copy) # after intervention with answer

        # Encoder
        for layer in self.layers:
            x = layer(x, y, mask, iod_mask)
            x_copy = layer(x_copy, y, mask_copy, iod_mask)

        # VLM
        if self.use_knowledge:
            inputs = self.tokenizer(knowledges, return_tensors="pt", add_special_tokens=False,
                                    truncation=True, max_length=20, padding='max_length')
            inputs.to(x.device)
            # print('know_embed shape: ', know_embed.shape)
            # original x shape is (bs, 36, 512)
            bs, hidden_size = x.shape[0], x.shape[-1]
            x = x.unsqueeze(1).repeat((1, 5, 1, 1)).view(bs*5, -1, hidden_size)
            # print('x repeated shape: ', x.shape)

            visual_token_type_ids = torch.ones(
                x.shape[:-1], dtype=torch.long).to(x.device)
            visual_attention_mask = torch.ones(
                x.shape[:-1], dtype=torch.float).to(x.device)

            ans_attention_mask = torch.ones_like(inputs['attention_mask'][:, 0].unsqueeze(dim=1))
            inputs_ans = {}
            inputs_ans.update({
                    "inputs_embeds": ans,
                    "attention_mask": ans_attention_mask,
                    "visual_embeds": x,
                    "visual_token_type_ids": visual_token_type_ids,
                    "visual_attention_mask": visual_attention_mask,
            })
            # inputs_kb = {}
            # inputs_kb.update({
            #     "inputs_embeds": inputs_embeds,
            #     "attention_mask": torch.cat([inputs['attention_mask'], torch.ones_like(inputs['attention_mask'][:, 0].unsqueeze(dim=1))]),
            #     "visual_embeds": x,
            #     "visual_token_type_ids": visual_token_type_ids,
            #     "visual_attention_mask": visual_attention_mask,
            # })
            inputs.update(
                {
                    "visual_embeds": x,
                    "visual_token_type_ids": visual_token_type_ids,
                    "visual_attention_mask": visual_attention_mask,
                }
            )
            outputs_ans_x = self.VLM(**inputs_ans)
            ans_x = outputs_ans_x.last_hidden_state
            topk_knowledge, topk_embed = self.selector(ans_x, self.knowledge_embed, self.knowledge_full)  # bs, topk, 768
            inputs_kb = {}
            inputs_kb.update({
                "input_embeds": torch.cat([topk_embed, ans], dim=1),
                "visual_embeds": x,
                "visual_token_type_ids": visual_token_type_ids,
                "visual_attention_mask": visual_attention_mask,
            })
            outputs = self.VLM(**inputs_kb)
            x = outputs.last_hidden_state
            mask = x.new_ones(x.shape[:2], dtype=torch.long).unsqueeze(-2)
            match_pred = self.sim_func(outputs_ans_x.pooler_output.unsqueeze(dim=1), topk_embed)
            # print(x.shape) # (bs, 36+20, 512)

        # print('Encoder x shape: ', x.shape, mask.shape)
        return self.norm(x), mask, self.norm(x_copy), mask_copy, match_pred

    def ans_intervention(self, X):
        ori_shape = X.shape
        X = X.reshape(-1, X.shape[-1])
        dic_z = self.ans_dict.cuda()
        prior = self.ans_prob.cuda()

        # print(X.shape, dic_z.shape)
        attention = torch.matmul(X, dic_z.t()) / (512 ** 0.5)
        attention = F.softmax(attention, 1)

        z_hat = attention.squeeze(1).unsqueeze(2) * dic_z.unsqueeze(0)

        z = torch.matmul(prior.unsqueeze(0), z_hat.squeeze(0)).squeeze(1)
        z = z.reshape(ori_shape)

        return z


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.norm = LayerNorm(size)
        self.src_attn = src_attn

    def forward(self, x, y, mask, iod_mask):
        "Follow Figure 1 (left) for connections."
        y = self.norm(y)
        x = self.sublayer[0](x, lambda x: self.src_attn(
            x, y, y, iod_mask) + self.self_attn(x, x, x, mask))
        # print(3, x.shape, y.shape)
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N, opt):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        # self.CI = Causual_intervention(opt)

    def forward(self, x, memory, src_mask, tgt_mask):
        # print('decoder forward before memory', memory.shape)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        # x = x + self.CI(x)
        # print('decoder forward after x', x.shape)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, opt):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        # print('DecoderLayer m', m.shape)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, None))
        return self.sublayer[2](x, self.feed_forward)


class Causual_intervention (nn.Module):
    def __init__(self, opt):
        super(Causual_intervention, self).__init__()
        self.z1 = nn.Parameter(torch.from_numpy(np.load(opt.lin_dict)).cuda().mean(
            dim=0).unsqueeze(dim=0).to(torch.float32).requires_grad_())
        self.z2 = nn.Parameter(torch.from_numpy(
            np.load(opt.visual_dict)).cuda().to(torch.float32).requires_grad_())
        self.z1_k = nn.Linear(opt.glove_dim, opt.input_encoding_size)
        self.z1_v = nn.Linear(opt.glove_dim, opt.input_encoding_size)
        self.z2_k = nn.Linear(opt.att_feat_size, opt.input_encoding_size)
        self.z2_v = nn.Linear(opt.att_feat_size, opt.input_encoding_size)
        self.v_q = nn.Linear(opt.input_encoding_size, opt.input_encoding_size)
        self.s_q = nn.Linear(opt.input_encoding_size, opt.input_encoding_size)

    def forward(self, x):
        z1 = self.z1_k(self.z1)
        z2 = self.z2_k(self.z2)
        att_S = torch.matmul(x, torch.t(z1)) / math.sqrt(x.size(-1))
        att_S = F.softmax(att_S, dim=-1)
        S = torch.matmul(att_S, self.z1_v(self.z1))
        att_V = torch.matmul(x, torch.t(z2)) / math.sqrt(x.size(-1))
        att_V = F.softmax(att_V, dim=-1)
        V = torch.matmul(att_V, self.z2_v(self.z2))
        att_res = self.v_q(V) + self.s_q(S)
        return att_res


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # print('MultiHeadedAttention 1', query.shape, key.shape, value.shape)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # print('MultiHeadedAttention 2', query.shape, key.shape, value.shape)
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, glove_dim, opt):
        super(Embeddings, self).__init__()
        # self.Glove_embedding = Glove_embedding(glove_dim, d_model, opt)
        self.embedding = nn.Embedding(29599, glove_dim)
        # print('num Embedding', self.embedding.num_embeddings)
        self.pro = nn.Linear(glove_dim, d_model)

    def forward(self, x):
        return self.pro(self.embedding(x))


class Glove_embedding(nn.Module):
    def __init__(self, glove_dim, d_model, opt):
        super(Glove_embedding, self).__init__()
        self.glove_dim = glove_dim
        self.dict = torch.from_numpy(
            np.load(opt.glove_embedding_dict)).cuda().to(torch.float32)
        self.pro = nn.Linear(glove_dim, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.to(torch.long).flatten()
        x = self.pro(torch.index_select(self.dict, 0, x).view(
            batch_size, -1, self.glove_dim))
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def factorization_loss(f_a, f_b):
    # empirical cross-correlation matrix
    f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0)+1e-6)
    f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0)+1e-6)
    c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c).pow_(2).mean()
    loss = on_diag + 0.005 * off_diag

    return loss


class KECVQG(AttModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6, glove_dim=300,
                   d_model=512, d_ff=2048, h=8, dropout=0.1, opt=None):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(attn), c(
                ff), dropout), N_enc, opt, self.tokenizer),
            Decoder(DecoderLayer(d_model, c(attn), c(
                attn), c(ff), dropout, opt), N_dec, opt),
            # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            lambda x: x,
            nn.Sequential(Embeddings(d_model, glove_dim, opt), c(position)),
            Generator(d_model, tgt_vocab))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt, tokenizer=None):
        super(KECVQG, self).__init__(opt)
        self.opt = opt
        self.tokenizer = tokenizer
        # self.config = yaml.load(open(opt.config_file))

        self.N_enc = getattr(opt, 'N_enc', opt.num_layers)
        self.N_dec = getattr(opt, 'N_dec', opt.num_layers)
        self.d_model = getattr(opt, 'd_model', opt.input_encoding_size)
        self.glove_dim = getattr(opt, 'glove_dim', opt.input_encoding_size)
        self.d_ff = getattr(opt, 'd_ff', opt.rnn_size)
        self.h = getattr(opt, 'num_att_heads', 8)
        self.dropout = getattr(opt, 'dropout', 0.1)

        delattr(self, 'att_embed')
        self.att_embed = nn.Sequential(*(
            ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
            (nn.Linear(self.att_feat_size, self.d_model),
             nn.ReLU(),
             nn.Dropout(self.drop_prob_lm)) +
            ((nn.BatchNorm1d(self.d_model),) if self.use_bn == 2 else ())))
        if opt.enable_VD:
            self.att_embed_iod = nn.Sequential(*(
                ((nn.BatchNorm1d(self.iod_feat_size + (opt.aug_feat_size if opt.enable_AFM > 0 else 0)),) if self.use_bn else ()) +
                (nn.Linear(self.iod_feat_size + (opt.aug_feat_size if opt.enable_AFM > 0 else 0), self.d_model),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.d_model),) if self.use_bn == 2 else ())))
        else:
            self.att_embed_iod = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size + (opt.aug_feat_size if opt.enable_AFM > 0 else 0)),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size + (opt.aug_feat_size if opt.enable_AFM > 0 else 0), self.d_model),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.d_model),) if self.use_bn == 2 else ())))

        delattr(self, 'embed')
        self.embed = lambda x: x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x: x
        delattr(self, 'logit')
        del self.ctx2att

        tgt_vocab = self.vocab_size + 1

        self.VD = VisualDeconfounder(2048)

        self.causal_factorization = nn.Linear(2048, opt.aug_feat_size)

        self.model = self.make_model(0, tgt_vocab,
                                     N_enc=self.N_enc,
                                     N_dec=self.N_dec,
                                     glove_dim=self.glove_dim,
                                     d_model=self.d_model,
                                     d_ff=self.d_ff,
                                     h=self.h,
                                     dropout=self.dropout, opt=self.opt)

    def logit(self, x):  # unsafe way
        return self.model.generator.proj(x)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, knowledges, fc_feats, att_feats, iod_feats, att_masks, iod_masks, answers):

        att_feats, iod_feats, seq, att_masks, iod_masks, seq_mask = self._prepare_feature_forward(
            att_feats, iod_feats, att_masks, iod_masks)
        memory, att_masks, _, _, _ = self.model.encode(
            knowledges, att_feats, iod_feats, att_masks, iod_masks, answers)

        return fc_feats[..., :0], att_feats[..., :0], memory, att_masks

    def _prepare_feature_forward(self, att_feats, iod_feats, att_masks=None, iod_masks=None, seq=None):

        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        iod_feats, iod_masks = self.clip_att(iod_feats, iod_masks)

        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        iod_feats = pack_wrapper(self.att_embed_iod, iod_feats, iod_masks)

        if att_masks is None:  # True
            att_masks = att_feats.new_ones(
                att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if iod_masks is None:
            iod_masks = iod_feats.new_ones(
                iod_feats.shape[:2], dtype=torch.long)
        iod_masks = iod_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            # seq = seq[:,:-1]
            seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)
            seq_mask[:, 0] = 1  # bos

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            seq_per_img = seq.shape[0] // att_feats.shape[0]
            if seq_per_img > 1:
                att_feats, att_masks, iod_feats, iod_masks = utils.repeat_tensors(seq_per_img,
                                                                                  [att_feats, att_masks,
                                                                                      iod_feats, iod_masks]
                                                                                  )
        else:
            seq_mask = None

        return att_feats, iod_feats, seq, att_masks, iod_masks, seq_mask

    def _forward(self, knowledges, aug_feats, att_feats, objects_id, boxes, seq, answers, att_masks=None, iod_masks=None):
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
        if answers.ndim == 3:
            answers = answers.reshape(-1, answers.shape[2])

        iod_feats = None, None
        if self.opt.enable_VD:
            iod_feats = self.VD(att_feats, boxes, objects_id)
            iod_feats = iod_feats.to(att_feats.device)
        else:
            iod_feats = att_feats.clone()

        loss_att = None
        if self.opt.enable_AFM:
            aug_feats = self.causal_factorization(aug_feats)
            aug_feats = aug_feats.reshape(-1, aug_feats.shape[-1])

            att_feats_tmp = self.causal_factorization(att_feats)
            att_feats_tmp = att_feats_tmp.reshape(aug_feats.shape)

            loss_att = factorization_loss(aug_feats, att_feats_tmp)
            iod_feats = torch.cat([iod_feats, aug_feats.reshape(
                iod_feats.shape[0], iod_feats.shape[1], -1)], dim=2)

        if self.training:
            att_feats, iod_feats, seq, att_masks, iod_masks, seq_mask = self._prepare_feature_forward(
                att_feats, iod_feats, att_masks, iod_masks, seq)

            out_know, out, match_pred = self.model(
                knowledges, att_feats, iod_feats, seq, answers, att_masks, iod_masks, seq_mask)
            # print('out_know', out_know.shape, out.shape)  # (bs, 21, 512)
            outputs_know = self.model.generator(out_know)
            outputs = self.model.generator(out)
            # print('output', outputs.shape) # (bs, 21, 29599)
            return outputs_know, outputs, loss_att, match_pred
        else:
            return None

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)

        out = self.model.decode(memory, mask, ys,
                                subsequent_mask(ys.size(1))
                                .to(memory.device))
        # print('shape ', out.shape, out[:, -1].shape)
        return out[:, -1], [ys.unsqueeze(0)]
