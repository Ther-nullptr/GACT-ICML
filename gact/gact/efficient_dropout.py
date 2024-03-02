"""
Low-Memory Dropout
==================

In this tutorial, you will write a memory-efficient implementation of dropout whose state
will be composed of a single int32 seed. This differs from more traditional implementations of dropout,
whose state is generally composed of a bit mask tensor of the same shape as the input.

In doing so, you will learn about:

* The limitations of naive implementations of Dropout with PyTorch.

* Parallel pseudo-random number generation in Triton.

"""

# %%
# Baseline
# --------
#
# The *dropout* operator was first introduced in [SRIVASTAVA2014]_ as a way to improve the performance
# of deep neural networks in low-data regime (i.e. regularization).
#
# It takes a vector as input and produces a vector of the same shape as output. Each scalar in the
# output has a probability :math:`p` of being changed to zero and otherwise it is copied from the input.
# This forces the network to perform well even when only :math:`1 - p` scalars from the input are available.
#
# At evaluation time we want to use the full power of the network so we set :math:`p=0`. Naively this would
# increase the norm of the output (which can be a bad thing, e.g. it can lead to artificial decrease
# in the output softmax temperature). To prevent this we multiply the output by :math:`\frac{1}{1 - p}`, which
# keeps the norm consistent regardless of the dropout probability.
#
# Let's first take a look at the baseline implementation.

import tabulate
import torch

import triton
import triton.language as tl

@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _seeded_dropout_backward(
    grad_out_ptr,
    grad_in_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from grad_out
    mask = offsets < n_elements
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    grad_out_keep = random > p
    # write-back
    grad_in = tl.where(grad_out_keep, grad_out / (1 - p), 0.0)
    tl.store(grad_in_ptr + offsets, grad_in, mask=mask)


class EfficientMemoryDropoutFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, seed):
        ctx.p = p
        ctx.seed = seed
        output = torch.empty_like(x)
        assert x.is_contiguous()
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        p, seed = ctx.p, ctx.seed
        grad_in = torch.empty_like(grad_out)
        assert grad_out.is_contiguous()
        n_elements = grad_out.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        _seeded_dropout_backward[grid](grad_out, grad_in, n_elements, p, seed, BLOCK_SIZE=1024)
        return grad_in, None, None
    

class EfficientMemoryDropout(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        # randomly generate a seed(per layer)
        self.seed = torch.randint(0, 2**32, (1,)).item()

    def forward(self, x):
        # notice that dropout works differently in training and evaluation mode
        if self.training:
            return EfficientMemoryDropoutFunc.apply(x, self.p, self.seed)
        else:
            return x
    

if __name__ == '__main__':
    # test the nn Module
    x = torch.randn(size=(2, 2, 2)).cuda()
    print(x)
    dropout = EfficientMemoryDropout(0.5).cuda()
    output = dropout(x)
    print(output)
