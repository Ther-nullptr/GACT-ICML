import torch
from gact.utils import get_dct_matrix, get_dqf_matrix

class DCTProcessor(torch.nn.Module):
  def __init__(self, quality=75, interpolation=1.):
    super(DCTProcessor, self).__init__()
    self.quality = quality
    self.quant_matrix = get_dqf_matrix(quality, flatten=True, interpolation=interpolation).to(torch.float16).to('cuda')
    self.dct_base = get_dct_matrix(int(64 * interpolation)).to(torch.float16).to('cuda')

  def forward(self, x):
    '''
    # The matrix is quantized by following algorithm:
    scaling_factor = 127 / (np.max(original_data))
    original_data = np.clip(np.round((original_data) * scaling_factor), -128, 127)
    # then, the following vector must be viewed as a K-dimension 64 matrix
    '''
    # DCT
    x = x.to(torch.float16)
    C = torch.matmul(self.dct_base, x) 
    # let quant_matrix's shape to match the shape of C
    len_C = len(C.shape)
    self.quant_matrix = self.quant_matrix.view([1] * (len_C - 2) + [-1] + [1])
    # quantize then dequantize
    quantized_C = torch.round(C / self.quant_matrix) * self.quant_matrix 
    # IDCT
    P = torch.round(torch.matmul(self.dct_base.T, quantized_C))
    return P


if __name__ == '__main__':
  dct_processor = DCTProcessor()
  x = torch.rand(2, 2, 64)
  scaling_factor = 127 / (torch.max(x, dim=0)[0])
  x = torch.clamp(torch.round((x) * scaling_factor), -128, 127)
  print(x)
  print(dct_processor(x))
