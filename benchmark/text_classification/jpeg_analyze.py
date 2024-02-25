import os
import torch
import torchvision

if __name__ == '__main__':
  activation_dir = '/home/yujin-wa20/projects/GACT-ICML/benchmark/text_classification/output_activations/lora'
  quality = 30
  # list all the files in the directory
  files = os.listdir(activation_dir)
  # iterate over all the files
  total_bytes_before_jpeg = 0
  total_bytes_after_jpeg = 0

  print(f'quality: {quality}')

  for file in files:
    # load the file
    activations = torch.load(os.path.join(activation_dir, file)).detach().cpu().to(torch.uint8)
    if len(activations.shape) > 2:
      input_shape_tmp = torch.Size((torch.prod(torch.tensor(activations.shape[:-1])).item(), activations.shape[-1]))
    else:
      input_shape_tmp = activations.shape
    
    print(f'input_shape_tmp: {input_shape_tmp}')
    activations = activations.view(input_shape_tmp).unsqueeze(0)

    bytes_before_jpeg = activations.element_size() * activations.nelement()
    total_bytes_before_jpeg += bytes_before_jpeg

    jpeg_activations = torchvision.io.encode_jpeg(activations, quality=quality)
    bytes_after_jpeg = jpeg_activations.element_size() * jpeg_activations.nelement()
    total_bytes_after_jpeg += bytes_after_jpeg

    print(f'File: {file}, Bytes before JPEG: {bytes_before_jpeg}, Bytes after JPEG: {bytes_after_jpeg}, compression ratio: {bytes_before_jpeg / bytes_after_jpeg}')

  print(f'Total bytes before JPEG: {total_bytes_before_jpeg}, Total bytes after JPEG: {total_bytes_after_jpeg}, Total compression ratio: {total_bytes_before_jpeg / total_bytes_after_jpeg}')
    
    