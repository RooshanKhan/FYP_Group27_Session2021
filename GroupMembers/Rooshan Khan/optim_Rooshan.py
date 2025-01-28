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
        # print(loss)
        # print(closure)
        '''Our Comments'''
        """self.param_groups is a list with only one element that is a dictionary. The coming for loop line acces the only element of the list."""
        for group in self.param_groups:     # Here group is a dictionary
            # print("Hello")
            # print(self.param_groups)
            # print(group)
            # print(group==self.param_groups)
            # Theta_t=group[""]
            # raise NotImplementedError
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]
                Lambda = group["weight_decay"]
                epsilon = group["eps"]
                Betas=group["betas"]
                Beta1=Betas[0]
                Beta2=Betas[1]
                """Initialization"""
                mt=torch.zeros(p.data.shape)
                vt=torch.zeros(p.data.shape)
                t=torch.zeros(p.data.shape)

                # print(group)
                # print ("state ",state)

                # print(p.data)
                # print(p)

                grad_zero=torch.zeros(grad.shape)
                # print(grad_zero,grad)
                while (torch.any(~(grad==grad_zero)).item()==False): # We use `tensor.item()` in Python to convert a 0-dim tensor to a number
                    print(torch.any(~(grad==grad_zero)).item()==False)
                    t=t+1
                    # print("p=",p)
                    # print("grad=",grad)
                    # print("Length of group['params']=",len(group["params"]))
                    # print("p=",p)
                    mt=Beta1*mt+(1-Beta1)*grad
                    vt=Beta2*vt+(1-Beta2)*grad*grad
                    # Less efficient version of the algorithm
                    
                    mt=mt/(1-Beta1**t)
                    vt=vt/(1-Beta2**t)
                    p.data=p.data-alpha*mt/(vt**(1/2)+epsilon)
                    
                    # Efficient version of the algorithm
                    
                    # alpha_t=(alpha*math.sqrt(1-Beta2**t))/(1-Beta2**t)
                    # p.data=p.data-alpha_t*mt/(vt**(1/2)+epsilon)
                    
                    #-----------------------------------
                    p.data=p.data-alpha*Lambda*p.data
                    loss=p.grad.data

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
                # raise NotImplementedError
        # print(self.param_groups,self.param_groups[0],self.param_groups[0]["params"],self.param_groups[0]["params"][0])
        # loss=self.param_groups[0]["params"][0].grad.data
        return loss