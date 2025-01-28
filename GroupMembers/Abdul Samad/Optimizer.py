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

        for group in self.param_groups:
            print("step")
            for p in group["params"]:
                print("p", p)
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]
                print("State", state)

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]
                eps = group["eps"]
                Lambda = group["weight_decay"]
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                
                

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
                if len(state)==0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    state["step"] = 0   
                
                m, v, t = state["m"], state["v"], state["step"]
                
                beta1,beta2 = group["betas"]
                
                m = ((m*beta1) + (grad*(1-beta1)))
                v = ((v*beta2) + (grad*grad*(1-beta2)))
                
                print("moment1", m)
                print("moment2", v)
                              
                t += 1
                
                bias_correction1 = (1- beta1**t)
                bias_correction2 = (1- beta2**t)
                
                #αt ← α · sqrt(1 − β2t) /(1 − β1t )
                
                step_size = alpha * (math.sqrt(bias_correction2))/(bias_correction1)
                print("step_size = ", step_size)
                # θt ← θt−1 − αt · mt /(sqrt(vt) + ϵ)
                denominator = (v.sqrt())+ eps
                p.data = p.data - ((step_size*m)/denominator)
                print("p.data = ",p.data)
                
                if Lambda != 0:
                    # θt ​=θt ​− α⋅λ⋅θ 
                    p.data = p.data - lr * Lambda * p.data
                    print(p.data)   

                
                state["m"], state["v"], state["step"] = m, v, t
                print("m: ", m)
                print("v: ", v)
                
                   
        return loss
