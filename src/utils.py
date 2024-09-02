import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AverageMeter(object):
  def __init__(self):
    self.val = None
    self.sum = None
    self.cnt = None
    self.avg = None
    self.ema = None
    self.initialized = False

  def update(self, val, n=1):
    if not self.initialized:
      self.initialize(val, n)
    else:
      self.add(val, n)

  def initialize(self, val, n):
    self.val = val
    self.sum = val * n
    self.cnt = n
    self.avg = val
    self.ema = val
    self.initialized = True

  def add(self, val, n):
    self.val = val
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
    self.ema = self.ema * 0.99 + self.val * 0.01


def inter_and_union(pred, mask, num_class):
  pred = np.asarray(pred, dtype=np.uint8).copy()
  mask = np.asarray(mask, dtype=np.uint8).copy()

  # 255 -> 0
  pred += 1
  mask += 1
  pred = pred * (mask > 0)

  inter = pred * (pred == mask)
  (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
  (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
  (area_mask, _) = np.histogram(mask, bins=num_class, range=(1, num_class))
  area_union = area_pred + area_mask - area_inter

  return (area_inter, area_union)


#Define trainform to images
def img_transform(trainsize= 256):
    train_transform = A.Compose([
    A.Rotate([-10,10]),
    A.RandomScale([0.5,2]),
    A.GaussianBlur(),
    A.HorizontalFlip(),
    A.Resize(width=trainsize, height=trainsize),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
    ])

    test_transform = A.Compose([
        A.Resize(width=trainsize, height=trainsize),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ToTensorV2(),
    ])
    return train_transform, test_transform