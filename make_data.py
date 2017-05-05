import sys
import os
from PIL import Image
import numpy as np
import json


def get_png_list():
    file_list = []
    for (root, dirs, files) in os.walk('.'):
        for filename in files:
            if len(filename) < 5 or filename[-4:] != '.png':
                continue
            path = os.path.join(root, filename).replace('\\', '/')
            file_list.append(path)
    return file_list


def convert(image_file):
    image = Image.open(image_file).resize((48, 48))
    if image.mode == 'RGB':
        image = image.convert('RGBA')
    return image


save_dir = '../data'
def make_image_set(file_list):
    os.mkdir(save_dir)
    # with open('dataset.txt', 'w') as f:
    for image_file in file_list:
        image = convert(image_file)
        name = image_file.split('/')[-1]
        image.save(save_dir + '/' + name)
        basename = name.split('.')[0]
        tmp = image.transpose(Image.ROTATE_180)
        tmp.save(save_dir + '/' + basename + '_rev.png')
        # label = name.split('_')[0]
        # data = {
        #     'image': image,
        #     'label': label
        # }
        # f.write(json.dumps(data) + '\n')


def make_test_image_set():
    test_path = '../test'
    test_save_dir = '../test_data'
    if os.path.exists(test_save_dir):
        print('test data already exists: skipped')
        return
    os.mkdir(test_save_dir)
    for image_file in os.listdir(test_path):
        path = os.path.join(test_path, image_file)
        image = convert(path)
        image.save(os.path.join(test_save_dir, image_file))


def train_data():
    with open('label.txt') as f:
        label_dict = json.load(f)

    with open('train.txt', 'w') as train_file:
        for filename in np.random.permutation(os.listdir(save_dir)):
            label = filename.split('_')[0]
            path = os.path.join(save_dir, filename)
            train_file.write('%s %d\n' % (path, label_dict[label]))


def set_testdata():
    test_dir = '../test_data/'
    jihai = [
        'chu', 'haku', 'hatsu', 'ton', 'nan', 'sha', 'pei'
    ]
    for filename in os.listdir(test_dir):
        label = filename.split('_')[0]
        if label in jihai:
            target = label
        else:
            target = label[0] + '/' + label[1:]
        old_path = os.path.join(test_dir, filename)
        new_path = os.path.join(target, filename)
        os.rename(old_path, new_path)


if __name__ == '__main__':
    if not os.path.exists(save_dir):
        print('making image set')
        file_list = get_png_list()
        make_image_set(file_list)

    print('making training txt')
    train_data()
    make_test_image_set()
    # set_testdata()
