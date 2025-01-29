from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        '''Our Comments'''
        """self.param_groups is a list with only one element that is a dictionary. The for loop in the next line acceses the only element of the list."""
        for group in self.param_groups:     # Here group is a dictionary

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # To complete this implementation:
                # 1. Update the first and second moments of the gradients.
                # 2. Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3. Update parameters (p.data).
                # 4. Apply weight decay after the main gradient-based updates.
                # Refer to the default project handout for more details.
                ### TODO

                m0=torch.zeros(p.data.shape)
                v0=torch.zeros(p.data.shape)
                t0=0

                if (len(state)==0):
                    state['t']=t0
                    state['mt']=m0
                    state['vt']=v0
                    self.state[p]=state
                # print("self.state = ",state)
                mt=state['mt']
                vt=state['vt']
                t=state['t']
                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]
                Lambda = group["weight_decay"]
                epsilon = group["eps"]
                Betas=group["betas"]
                Beta1=Betas[0]
                Beta2=Betas[1]

                t=t+1
                mt=Beta1*mt+(1-Beta1)*grad
                vt=Beta2*vt+(1-Beta2)*grad*grad

                # Less efficient version of the algorithm
                #--------------------------------------
                # mt_hat=mt/(1-Beta1**t)
                # vt_hat=vt/(1-Beta2**t)
                # p.data=p.data-alpha*mt_hat/(vt_hat**(1/2)+epsilon)
                #--------------------------------------
                # Efficient version of the algorithm
                #--------------------------------------
                alpha_t=(alpha*math.sqrt(1-Beta2**t))/(1-Beta1**t)
                p.data=p.data-alpha_t*mt/(vt**(1/2)+epsilon)
                #-----------------------------------
                p.data=p.data-alpha*Lambda*p.data

                state['t']=t
                state['mt']=mt
                state['vt']=vt
                # self.state[p]=state   # This line is redundant because updates to state are directly reflected to self.stae[p]




        # print("Loss = ",p.data,"    Closure = ",closure)
        # print (self.state[p])
        return loss