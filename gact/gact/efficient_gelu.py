import torch
import torch.nn.functional as F
from gact.dct_processor import DCTProcessor
from gact.jpeg_processor import JPEGProcessor
from gact.memory_efficient_function import per_block_quantization, per_block_dequantization, dct_compression, jpeg_compression

class EfficientMemoryGELUFunc(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, compress_type, jpeg_processor, dct_processor):
    result = F.gelu(x)
    ctx.needs_inputs_grad = x.requires_grad
    input_shape = x.shape
    ctx.input_shape = input_shape
    ctx.compress_type = compress_type

    x, quant_state = per_block_quantization(x, input_shape)
    ctx.quant_state = quant_state

    if compress_type == 'JPEG':
        x = jpeg_compression(x, input_shape, jpeg_processor)

    elif compress_type == 'DCT':
        x = dct_compression(x, input_shape, dct_processor)

    ctx.save_for_backward(x)
    return result

  @staticmethod
  def backward(ctx, grad_output):
    x, = ctx.saved_tensors
    quant_state = ctx.quant_state
    input_shape = ctx.input_shape

    grad_input = None

    if ctx.needs_inputs_grad:
      # dequantize the cached activation
      if ctx.compress_type == 'JPEG':
          pass
      
      elif ctx.compress_type == 'DCT':
          x = per_block_dequantization(x, input_shape, quant_state)

      grad_input = None
      if ctx.needs_inputs_grad:
        exp_term = torch.exp(-0.5 * x * x)
        grad_input = F.gelu(x) + 0.5 * x * exp_term / 1.41421
        grad_input = grad_input * grad_output

    return grad_input, None, None, None
  

class EfficientMemoryGELU(torch.nn.Module):
  def __init__(self, compress_type: str = "JPEG", compress_quality: int = 50):
    super(EfficientMemoryGELU, self).__init__()
    self.compress_type = compress_type
    self.compress_quality = compress_quality
    self.jpeg_processor = None # JPEGProcessor(quality=compress_quality)
    self.dct_processor = DCTProcessor(quality=compress_quality)

  def forward(self, input):
    return EfficientMemoryGELUFunc.apply(
      input,
      self.compress_type,
      self.jpeg_processor,
      self.dct_processor
    )