import os
import sys
import cPickle as pickle
import numpy as np
import tensorflow as tf

def _read_data(data_path, train_files, resolution):
  """Reads CIFAR-100 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  images, labels = [], []
  for file_name in train_files:
    print file_name
    full_name = os.path.join(data_path, file_name)
    with open(full_name) as finp:
      data = pickle.load(finp)
      batch_images = data["data"].astype(np.float32) / 255.0
      batch_labels = np.array(data["fine_labels"], dtype=np.int32)
      images.append(batch_images)
      labels.append(batch_labels)
  images = np.concatenate(images, axis=0)
  labels = np.concatenate(labels, axis=0)
  images = np.reshape(images, [-1, 3, 32, 32])
  images = np.transpose(images, [0, 2, 3, 1])

  return images, labels


def read_data(data_path, resolution=32, num_valids=5000):
  print "-" * 80
  print "Reading data"

  images, labels = {}, {}

  train_files = [
    "train",
  ]
  test_file = [
    "test",
  ]
  images["train"], labels["train"] = _read_data(data_path, train_files, resolution)

  if num_valids:
    images["valid"] = images["train"][-num_valids:]
    labels["valid"] = labels["train"][-num_valids:]

    images["train"] = images["train"][:-num_valids]
    labels["train"] = labels["train"][:-num_valids]
  else:
    images["valid"], labels["valid"] = None, None

  images["test"], labels["test"] = _read_data(data_path, test_file, resolution)

  print "Prepropcess: [subtract mean], [divide std]"
  mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
  std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

  print "mean: {}".format(np.reshape(mean * 255.0, [-1]))
  print "std: {}".format(np.reshape(std * 255.0, [-1]))

  images["train"] = (images["train"] - mean) / std
  if num_valids:
    images["valid"] = (images["valid"] - mean) / std
  images["test"] = (images["test"] - mean) / std

  return images, labels

