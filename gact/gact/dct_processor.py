import torch
from gact.utils import get_dct_matrix, get_dqf_matrix

class DCTProcessor(torch.nn.Module):
  def __init__(self, quality=75):
    super(DCTProcessor, self).__init__()
    self.quality = quality
    self.quant_matrix = get_dqf_matrix(quality, flatten=True).to('cuda')
    self.dct_base = get_dct_matrix(64).to('cuda')

  def forward(self, x):
    '''
    # The matrix is quantized by following algorithm:
    scaling_factor = 127 / (np.max(original_data))
    original_data = np.clip(np.round((original_data) * scaling_factor), -128, 127)
    # then, the following vector must be viewed as a K-dimension 64 matrix
    '''
    # DCT
    x = x.to(torch.float32)
    C = torch.matmul(self.dct_base, x) 
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
