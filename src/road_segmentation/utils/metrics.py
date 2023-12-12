from torch.utils.data import DataLoader
import torch.nn as nn

import torch

def compute_f1(dataset, model, device, threshold=0.25):
  model.eval()
  
  with torch.no_grad():  
    dl = DataLoader(dataset, batch_size=32)
    tp, fp, fn = 0, 0, 0
    
    for data, target in dl:
      data, target = data.to(device), target.to(device)
      output = model.forward(data)
      output = nn.Sigmoid()(output)
      
      output = (output > threshold).flatten()
      target = (target > threshold).flatten()
      
      tp += (output & target).sum()
      fp += (output & ~target).sum()
      fn += (~output & target).sum()
    
    f1 = 2 * tp / (2 * tp + fp + fn)
  
  model.train()
  
  return f1