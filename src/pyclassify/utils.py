import os
import yaml

def distance(point1: list[float], point2: list[float]):
    square_distance = 0
    for x1, x2 in zip(point1, point2):
        square_distance += (x1 - x2) ** 2
    return square_distance

def majority_vote(neighbors: list[int]):
    # freq_dict = {x: neighbors.count(x) for x in set(neighbors)}
    # max_freq_val = max(freq_dict.values())
    # max_freq_key = list(freq_dict.keys())[list(freq_dict.values()).index(max_freq_val)]
    # return max_freq_key
    freq_class_dict = {neighbors.count(x): x for x in set(neighbors)}
    max_freq = max(freq_class_dict.keys())
    max_freq_class = freq_class_dict.get(max_freq)
    return max_freq_class

def read_config(file):
   filepath = os.path.abspath(f'{file}.yaml')
   with open(filepath, 'r') as stream:
      kwargs = yaml.safe_load(stream)
   return kwargs

def read_file(filepath):
    def convert_label(label):
        if label == 'g\n':
            return 0
        if label == 'b\n':
            return 1
        else:
            raise ValueError("Invalid label")
    with open(filepath) as fp:
        lines = fp.readlines()
    samples, labels = [], []
    for line in lines:
        tmp = line.split(',')
        samples.append([float(x) for x in tmp[:-1]])
        labels.append(convert_label(tmp[-1]))
    return samples, labels