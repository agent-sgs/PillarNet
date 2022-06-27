import torch
import torch.nn as nn
from . import pillar_cuda
from torch.autograd import Function, Variable


class ScatterMaxFunction(Function):
    @staticmethod
    def forward(ctx, src:torch.Tensor, index:torch.Tensor, M:int):
        """
        Args:
            src: (C, L)
            index: (L, )
        Returns:
            out: (C, M)
        """
        assert index.is_contiguous()
        assert src.is_contiguous()
        C, L = src.size()

        # import time
        # torch.cuda.synchronize()
        # start = time.time()

        arg = torch.full([C, M], -1, dtype=index.dtype, device=index.device, requires_grad=False)
        out = src.new_zeros([C, M])
        pillar_cuda.scatter_max_wrapper(index, src, arg, out)

        # torch.cuda.synchronize()
        # end = time.time()
        # print(end - start)

        ctx.for_backwards = (C, L, arg)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        C, L, arg = ctx.for_backwards
        grad_src = Variable(torch.cuda.FloatTensor(C, L).zero_())
        grad_out_data = grad_out.data.contiguous()

        pillar_cuda.scatter_max_grad_wrapper(arg, grad_out_data, grad_src)
        return grad_src, None, None

scatter_max = ScatterMaxFunction.apply


