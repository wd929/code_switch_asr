import math

import torch
from numba import cuda

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

GPU_RNNT_THREAD_SIZE = 256


@cuda.jit()
def update_hyps(last_label, k_is_blank, v, hyp2b, hyp2t, scores, ys, m):
    b0 = cuda.blockIdx.x
    b = hyp2b[cuda.blockIdx.x]

    if k_is_blank[b0] == 0:
        ys[b * m] = ys[b * m] + 1


#    length = ys[b * m]
#    ys[b * m +length] = last_label[b0]


# @cuda.jit()
# def copy_dec_states(dec_states, hidden):
#  b = cuda.blockIdx.x
#  dec_states[b, 0] = hidden[0, b]
