import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 

class LinearClassifier(nn.Module):  
  def __init__(self, encoding_size):
    super(LinearClassifier, self).__init__()
    self.linear = nn.Linear(encoding_size, encoding_size)

  def forward(self, x):
    x = self.linear(x)
    x = torch.tanh(x)
    return x

class LinearCombine(nn.Module):
  def __init__(self, encoding_size):
    super().__init__()
    self.linear = nn.Linear(encoding_size, 27) # 27 is the number of genres

    self.sigmoid = nn.Sigmoid()

  def forward(self, x, y):
    x = self.linear(torch.cat((x, y), dim=1))
    x = self.sigmoid(x)
    return x


class Gated_MultiModal_Unit():

  def __init__(self, img_model, text_model):
    super().__init__()
    self.bert_model = text_model
    self.resnet_model = img_model
    self.hv_gate = LinearClassifier(27) # num of categories
    self.ht_gate = LinearClassifier(27) # num of categories
    self.z_gate = LinearCombine(54) # 2* num of categories, for data fusion



  def forward(self, mode,imgs, texts, text_masks):
    if (mode == 'train'):
      self.hv_gate.train()
      self.ht_gate.train()
      self.z_gate.train()

    else:
      self.hv_gate.eval()
      self.ht_gate.eval()
      self.z_gate.eval()

    self.resnet_model.eval()
    self.bert_model.eval()
    # self.resnet_model.cuda()
    # self.bert_model.cuda()
    img_pred = self.resnet_model(imgs)
    text_pred = self.bert_model(texts, text_masks).logits

    
    # self.hv_gate.cuda()
    # self.ht_gate.cuda()
    # self.z_gate.cuda()


    h_v = self.hv_gate(img_pred)
    h_t = self.ht_gate(text_pred)

    z = self.z_gate(img_pred, text_pred)

    h = (torch.mul(z, h_v)) + (torch.mul((1-z), h_t))

    predictions = h
    return predictions