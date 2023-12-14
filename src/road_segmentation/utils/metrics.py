import torch
from torch.utils.data import DataLoader
import torch.nn as nn

def compute_f1(dataset, model, device, threshold=0.25):
  model.eval()
  
  dl = DataLoader(dataset, batch_size=32)
  tp, fp, fn, tn = 0, 0, 0, 0

  with torch.no_grad():
    for data, target in dl:
      data, target = data.to(device), target.to(device)
      output = model.forward(data)
      output = nn.Sigmoid()(output)
      
      output = (output > threshold).flatten()
      target = (target > threshold).flatten()
      
      tp += (output & target).sum().item()
      fp += (output & ~target).sum().item()
      fn += (~output & target).sum().item()
      tn += (~output & ~target).sum().item()
  
  f1 = 2 * tp / (2 * tp + fp + fn)
  acc = (tp + tn) / (tp + tn + fp + fn)
  
  model.train()
  
  return acc, f1