import os
import torch
import torchvision
import numpy as np


def generate_test_train_text_file(test_train_ratio, root_dir):
    classes = os.listdir(root_dir)
    train_images = []
    test_images = []
    for category in classes:
        images = os.listdir(os.path.join(root_dir, category))
        images = [os.path.join(category, i) for i in images]
        train_images+=list(np.random.choice(np.array(images),int(test_train_ratio*len(images)),replace=False))
        test_images+=list(set(images).difference(set(train_images)))
    with open(os.path.join(root_dir, 'train.txt'), 'w') as f:
        for item in train_images:
            f.write("%s\n" % item)
    with open(os.path.join(root_dir, 'validation.txt'), 'w') as f:
        for item in test_images:
            f.write("%s\n" % item)
    with open(os.path.join(root_dir, 'classes.txt'), 'w') as f:
        for i, item in enumerate(classes):
            f.write("{0} {1} \n".format(i, item))


def get_labels(path_to_classes):
    classes ={}
    with open(path_to_classes) as f:
        c = f.readlines()
    for x in c:
        classes[x.strip().split(" ")[1]] =  int(x.strip().split(" ")[0])
    return classes


def get_file_names_from_text(path):
    with open(path) as f:
        files = f.readlines()
    files = [x.strip() for x in files]
    return files

def rewrite_text_file(data_root,classes_path):
    classes = list(get_labels(os.path.join(data_root, classes_path)).keys())
    training_file_names = []
    for c in classes:
        training_file_names += [os.path.join(c, p) for p in os.listdir(os.path.join(data_root, c))]
    open(os.path.join(data_root, 'train.txt'), 'w').close()
    with open(os.path.join(data_root, 'train.txt'), 'w') as f:
        for item in training_file_names:
            f.write("%s\n" % item)
