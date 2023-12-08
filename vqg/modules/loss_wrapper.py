import torch
from . import losses
import torch.nn as nn
import torch.nn.functional as F

# from ..utils.rewards import init_scorer


class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model

        if opt.label_smoothing > 0: # default is FALSE
            self.crit = losses.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = losses.LanguageModelCriterion()
        self.rl_crit = losses.RewardCriterion(opt)

        self.add_reader_loss = False
        if self.opt.loss_eff > 0:
            self.add_reader_loss = True
            self.loss_eff = PlainLoss('ce')
            self.sim_func = nn.CosineSimilarity(dim=-1)
            self.retriever_loss = nn.MSELoss()
            self.select_loss = nn.MSELoss()

    def forward(self, knowledges, aug_feats, att_feats, objects_id, boxes, labels, answers, masks, att_masks, iod_masks, gts, gt_indices, sc_flag):
        out = {}
        output_know, output, loss_att, match_pred = self.model(knowledges, aug_feats, att_feats, objects_id, boxes, labels[..., :-1], answers, att_masks, iod_masks)
        loss = self.crit(output_know, labels[..., 1:], masks[..., 1:])

        if self.add_reader_loss:
            _labels = output.new_zeros(output.shape, dtype=labels.dtype)
            sources = labels.new_ones(labels.shape)
            _labels.scatter_(dim=2, index=labels, src=sources)
            _labels[:, :, 0 ] =0

            vl_loss = self.loss_eff(output, _labels)
            vlk_loss = self.loss_eff(output_know, _labels)

            effecting, match_label = self.labeling(vlk_loss, vl_loss)
            loss_eff = (vlk_loss * effecting)
            loss_sel = self.select_loss(match_pred, match_label)
            out['loss_eff'] = loss_eff.mean() * self.opt.lambda1
            out['loss_sel'] = loss_sel.mean() * self.opt.lambda2
        if loss_att is not None:
            out['loss_att'] = loss_att
        
        out['loss'] = loss
        return out
    

    # @torch.no_grad()
    def labeling(self, vlk_loss, vl_loss):
        """ Make labels for both reader and retriever. """
        diff = vl_loss - vlk_loss
        label_retriever = torch.tanh(diff)
        label_reader = torch.sigmoid(diff)
        return label_reader, label_retriever


def CE_loss(logits, labels, weights):
    """ Modified cross entropy loss. """
    nll = F.log_softmax(logits, dim=-1)
    loss = - (nll * labels).sum(dim=-1)
    if weights is not None:
        loss = loss * weights
    return loss


class PlainLoss(nn.Module):
    def __init__(self, loss_type) -> None:
        super().__init__()
        self.loss_type = loss_type

    def forward(self, logits, labels, weights=None):
        if self.loss_type == 'ce':
            loss = CE_loss(logits, labels, weights)
        else: # bce
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            loss = loss.mean() if weights is None else (loss * weights).mean()
            loss *= labels.size(1)
        return loss