import chainer
import chainer.functions as F
import chainer.links as L


class Alex(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def __init__(self):
        super(Alex, self).__init__(
            # conv1=L.Convolution2D(3,  96, 2, stride=2), # -> 24
            # conv2=L.Convolution2D(96, 256,  4, pad=2), # -> 25 -> 12
            # conv3=L.Convolution2D(256, 384,  3, pad=1), # -> 12
            # conv4=L.Convolution2D(384, 384,  3, pad=1),
            # conv5=L.Convolution2D(384, 256,  3, pad=1), # -> 12 -> 6
            # fc6=L.Linear(9216, 4096),
            # fc7=L.Linear(4096, 1024),
            # fc8=L.Linear(1024, 37),
            conv1=L.Convolution2D(3,  64, 2, stride=2), # -> 24
            conv2=L.Convolution2D(64, 128,  4, pad=2), # -> 25 -> 12
            conv3=L.Convolution2D(128, 256,  3, pad=1), # -> 12
            conv4=L.Convolution2D(256, 256,  3, pad=1),
            conv5=L.Convolution2D(256, 128,  3, pad=1), # -> 12 -> 6
            fc6=L.Linear(4608, 1024),
            fc7=L.Linear(1024, 256),
            fc8=L.Linear(256, 37),
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x):
        # x = chainer.Variable(x_data)
        # t = chainer.Variable(t_data)
        # self.clear()
        h = F.relu(F.local_response_normalization(self.conv1(x)))
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 2, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)

        # self.loss = F.softmax_cross_entropy(h, t)
        # self.accuracy = F.accuracy(h, t)
        # return self.loss
        return h

    def predict(self, x_data):
        # x_data: array of image set
        x = chainer.Variable(x_data)
        h = F.relu(F.local_response_normalization(self.conv1(x)))
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 2, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)

        return F.softmax(h)

