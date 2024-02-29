import torch
from gact.utils import get_dct_matrix, get_dqf_matrix

class JPEGProcessor(torch.nn.Module):
  def __init__(self, quality=75):
    super(JPEGProcessor, self).__init__()
    self.quality = quality
    self.quant_matrix = get_dqf_matrix(quality, flatten=False)
    self.dct_base = get_dct_matrix(8)

  def forward(self, x):
    '''
    # The matrix is quantized by following algorithm:
    scaling_factor = 127 / (np.max(original_data))
    original_data = np.clip(np.round((original_data) * scaling_factor), -128, 127)
    # then, the following vector must be viewed as 8x8 matrix
    '''
    # 2D DCT
    C = torch.matmul(torch.matmul(self.dct_base, x), self.dct_base.T)
    # quantize then dequantize
    quantized_C = torch.round(C / self.quant_matrix) * self.quant_matrix 
    # IDCT
    P = torch.round(torch.matmul(torch.matmul(self.dct_base.T, quantized_C), self.dct_base))

    return P


if __name__ == '__main__':
  jpeg_processor = JPEGProcessor()
  x = torch.rand(2, 8, 8)
  scaling_factor = 127 / (torch.max(x, dim=0)[0])
  x = torch.clamp(torch.round((x) * scaling_factor), -128, 127)
  print(x)
  print(jpeg_processor(x))