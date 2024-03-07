import torch
from gact.utils import get_dct_matrix, get_dqf_matrix, get_walsh_matrix

class JPEGProcessor(torch.nn.Module):
  def __init__(self, quality=75):
    super(JPEGProcessor, self).__init__()
    self.quality = quality
    self.quant_matrix = get_dqf_matrix(quality, flatten=False).to(torch.float16).to('cuda')
    self.dct_base = get_dct_matrix(8).to(torch.float16).to('cuda')

  def forward(self, x):
    # '''
    # # The matrix is quantized by following algorithm:
    # scaling_factor = 127 / (np.max(original_data))
    # original_data = np.clip(np.round((original_data) * scaling_factor), -128, 127)
    # # then, the following vector must be viewed as 8x8 matrix
    # '''
    # 2D DCT
    x = x.to(torch.float16)
    C = torch.matmul(torch.matmul(self.dct_base, x), self.dct_base.T) # (8, 8) x (32, 16, 96, 8, 8) x (8, 8)
    # quantize then dequantize
    quantized_C = torch.round(C / self.quant_matrix) * self.quant_matrix 
    # IDCT
    P = torch.round(torch.matmul(torch.matmul(self.dct_base.T, quantized_C), self.dct_base))

    return P
  
  def forward_cpu(self, x):
    self.dct_base = self.dct_base.to('cpu').to(torch.float32)
    self.quant_matrix = self.quant_matrix.to('cpu').to(torch.float32)
    # 2D DCT
    x = x.to('cpu').to(torch.float32)
    C = torch.matmul(torch.matmul(self.dct_base, x), self.dct_base.T)
    # quantize then dequantize
    quantized_C = torch.round(C / self.quant_matrix) * self.quant_matrix
    # IDCT  
    P = torch.round(torch.matmul(torch.matmul(self.dct_base.T, quantized_C), self.dct_base))
    return P
  
  def forward_jpeg(self, x):
    self.dct_base = self.dct_base.to('cpu').to(torch.float32)
    self.quant_matrix = self.quant_matrix.to('cpu').to(torch.float32)
    # 2D DCT
    x = x.to('cpu').to(torch.float32)
    C = torch.matmul(torch.matmul(self.dct_base, x), self.dct_base.T) # (8, 8) x (32, 16, 96, 8, 8) x (8, 8)
    # quantize then dequantize
    quantized_C = torch.round(C / self.quant_matrix) * self.quant_matrix 
    return quantized_C
  
  def forward_jpeg_no_quant(self, x):
    self.dct_base = self.dct_base.to('cpu').to(torch.float32)
    self.quant_matrix = self.quant_matrix.to('cpu').to(torch.float32)
    x = x.to('cpu').to(torch.float32)
    # 2D DCT
    x = x.to(torch.float32)
    C = torch.matmul(torch.matmul(self.dct_base, x), self.dct_base.T) 
    return C


if __name__ == '__main__':
  jpeg_processor = JPEGProcessor()
  x = torch.rand(2, 8, 8)
  scaling_factor = 127 / (torch.max(x, dim=0)[0])
  x = torch.clamp(torch.round((x) * scaling_factor), -128, 127)
  print(x)
  print(jpeg_processor(x))