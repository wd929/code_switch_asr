import operator
from dataclasses import dataclass
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType
from nemo.core.utils.numba_utils import NUMBA_INSTALLATION_MESSAGE
from nemo.utils import logging, model_utils

try:
    import warprnnt_pytorch as warprnnt

    WARP_RNNT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    WARP_RNNT_AVAILABLE = False

try:
    from nemo.collections.asr.parts.numba.rnnt_loss import MultiblankRNNTLossNumba, RNNTLossNumba

    NUMBA_RNNT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NUMBA_RNNT_AVAILABLE = False


WARP_RNNT_INSTALLATION_MESSAGE = (
    "Could not import `warprnnt_pytorch`.\n"
    "Please visit https://github.com/HawkAaron/warp-transducer "
    "and follow the steps in the readme to build and install the "
    "pytorch bindings for RNNT Loss, or use the provided docker "
    "container that supports RNN-T loss."
)

if __name__ == "__main__":
    B, T, U, V = 1, 11, 7, 5
    B, T, U, V = 1, 2, 2, 2  # V is number of non blank labels
    B, T, U, V = 1, 3, 3, 5  # V is number of non blank labels
    B, T, U, V = 8, 264, 32, 128  # V is number of non blank labels

    big_blank_durations = [2, 3, 4, 5, 6, 7, 8]
    big_blank_durations = [2]
    #    big_blank_durations = []
    sigma = 0.1
    args = {}
    args['big_blank_durations'] = big_blank_durations
    args['sigma'] = sigma

    args2 = {}
    args2['big_blank_durations'] = big_blank_durations
    args2['sigma'] = sigma

    Loss = RNNTLoss(V, reduction='mean_batch', loss_name='multiblank_rnnt_pytorch', loss_kwargs=args)
    Loss2 = (
        RNNTLoss(V, reduction='mean_batch', loss_name='multiblank_rnnt', loss_kwargs=args2)
        if len(big_blank_durations) > 0
        else RNNTLoss(V, reduction='mean_batch', loss_name='warprnnt_numba', loss_kwargs=args2)
    )

    for t in range(22):
        acts = torch.rand([B, T, U, V + 1 + len(big_blank_durations)]) - 0.5
        acts = torch.nn.Parameter(acts * 5, requires_grad=True)

        labels = torch.randint(low=0, high=V - 1, size=[B, U])
        act_lens = torch.randint(low=1, high=T + 1, size=[B])
        label_lens = torch.randint(low=1, high=U + 1, size=[B]) - 1
        act_lens[0] = T
        label_lens[0] = U - 1
        logits = acts

        logits = logits.cuda()
        labels = labels.cuda()
        act_lens = act_lens.cuda()
        label_lens = label_lens.cuda()

        labels = labels.contiguous()

        loss = Loss(log_probs=logits, targets=labels, input_lengths=act_lens, target_lengths=label_lens)
        loss = torch.mean(loss)
        loss.backward()
        grad1 = torch.clone(acts.grad)
        acts.grad *= 0.0

        loss2 = Loss2(log_probs=logits, targets=labels, input_lengths=act_lens, target_lengths=label_lens)

        loss2.backward()

        print("loss diff", float(loss - loss2), float(loss), float(loss2))
        print("grad norm diff per element", float(torch.norm(acts.grad - grad1)))
