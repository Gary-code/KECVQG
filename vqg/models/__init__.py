from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import torch


from .AttModel import *

from .TransformersKECVQG import KECVQG


def setup(opt, tokenizer=None):
    if opt.vqg_model in ['fc', 'show_tell']:
        print('Warning: %s model is mostly deprecated; many new features are not supported.' %opt.vqg_model)
        if opt.vqg_model == 'fc':
            print('Use newfc instead of fc')

    elif opt.vqg_model == 'KECVQG':
         model = KECVQG(opt, tokenizer)
    else:
        raise Exception("VQG model not supported: {}".format(opt.vqg_model))

    return model
