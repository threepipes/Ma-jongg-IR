import os
import sys
import numpy as np
from PIL import Image

import matplotlib
import chainer
import chainer.cuda as ccuda
import chainer.functions as F
import chainer.links as L
from chainer import training, serializers
from chainer.training import extensions
import cupy

from model import Alex

matplotlib.use('Agg')

def convert(image, xp):
    if image.mode == 'RGB':
        image = image.convert('RGBA')
    pixels = xp.asarray(image).astype(xp.float32)
    pixels = pixels[:, :, ::-1]
    pixels = pixels.transpose(2, 0, 1)
    return pixels[1:]


def label2vec(label_list, n):
    vec_list = []
    for label in label_list:
        vec = [0] * n
        vec[label] = 1
        vec_list.append(vec)
    return vec_list


def load_test(path):
    mean = load_mean()
    data_list = []
    for filename in sorted(os.listdir(path)):
        img_path = os.path.join(path, filename)
        data = convert(Image.open(img_path), np)
        data_list.append((data - mean) / 255)
    return np.asarray(data_list)


def load(path, xp):
    mean = load_mean(gpu=True)
    data_list = []
    label_list = []
    label_size = 0
    with open('train.txt') as f:
        for row in f:
            line = row.split(' ')
            filename = line[0]
            label = int(line[1])
            target = os.path.join(path, filename)
            data = convert(Image.open(target), xp)
            # data_list.append(data / 255)
            data_list.append((data - mean) / 255)
            label_list.append(label)
            label_size = max(label_size, label + 1)
    # label_list = label2vec(label_list, label_size)
    result = []
    for data, label in zip(data_list, label_list):
        result.append((
            xp.asarray(data).astype(xp.float32),
            xp.asarray(label) # np.asarray(label).astype(np.int32)
        ))
    return result


def load_mean(gpu=False):
    mean = np.load('mean.npy')
    mean = mean[::-1][1:]
    if gpu:
        return chainer.cuda.to_gpu(mean)
    else:
        return mean


def make_batch(train_data):
    sub_list = train_data[:10]
    x = [data[0] for data in sub_list]
    t = [data[1] for data in sub_list]
    return np.asarray(x), np.asarray(t)


def train():
    if len(sys.argv) > 1:
        gpu = int(sys.argv[1])
    else:
        gpu = -1
    train_data = load('../data', cupy)

    model = Alex()
    classifier = L.Classifier(model)
    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()  # Make a specified GPU current
        model.to_gpu(gpu)  # Copy the model to the GPU
    # serializers.load_npz('my.model', model)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(classifier)

    train_size = int(len(train_data) * 0.8)
    test_size = len(train_data) - train_size

    train_iter = chainer.iterators.SerialIterator(train_data[:train_size], 20)
    test_iter = chainer.iterators.SerialIterator(train_data[train_size:], 20,
                                                 repeat=False, shuffle=False)
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    epoch = 200
    trainer = training.Trainer(updater, (epoch, 'epoch'), 'result')

    trainer.extend(extensions.Evaluator(test_iter, classifier))
    trainer.extend(extensions.dump_graph('main/loss'))
    frequency = epoch
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.LogReport())

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar(update_interval=100))

    print('begin train')
    # Run the training
    trainer.run()
    # alex = Alex()
    # x, t = make_batch(train_data)
    # loss = alex(x, t)
    # print(loss.data)
    serializers.save_npz('my_3.model', model)

    # train, test = chainer.datasets.get_mnist()


def predict():
    model = Alex()
    # load
    serializers.load_npz('my_3.model', model)
    test_data = load_test('../test_data')

    import json
    with open('label_table.txt') as f:
        labels = json.load(f)

    print('start predict')
    result = model.predict(test_data)

    result_list = list(result.data)
    label_result = []
    for idx, row in enumerate(result_list):
        tuples = [(prob, i) for i, prob in enumerate(row)]
        res = sorted(tuples)[-4:]
        print('image %d:' % idx)
        for r in res:
            print('%s\t%f' % (labels[r[1]], r[0]))
        label_result.append(labels[res[-1][1]])

    path = '../test_data/'
    for label, filename in zip(label_result, sorted(os.listdir(path))):
        os.rename(path + filename, path + label + '_' + filename)


if __name__ == '__main__':
    # train()
    predict()
