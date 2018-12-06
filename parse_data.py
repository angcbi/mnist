import struct

import numpy as np
from matplotlib import pyplot as plt

train_x_file = './source_mnist/train-images.idx3-ubyte'
train_y_file = './source_mnist/train-labels.idx1-ubyte'
test_x_file = './source_mnist/t10k-images.idx3-ubyte'
text_y_file = './source_mnist/t10k-labels.idx1-ubyte'

np_train_x_file = './data/train'
np_train_y_file = './data/train_label'
np_test_x_file = './data/test'
np_test_y_file = './data/test_label'


# parse train file
def parse_train_x_file():
    with open(train_x_file, 'rb') as file:
        desc_format = struct.Struct('>iiii')
        data_format = struct.Struct('>784B')

        # parse 16 byte
        magic, total, rows, cols = desc_format.unpack_from(file.read(desc_format.size))
        print(magic, total, rows, cols)

        # parse data
        mnist_list = []
        for i in range(total):
            # parse 784 byte
            read_data = file.read(data_format.size)
            data = data_format.unpack_from(read_data)
            mnist_list.append(data)

        train_x = np.array(mnist_list)
        np.save(np_train_x_file, train_x)

        # show first image, reshape 28*28
        fig = plt.figure()
        plt.imshow(train_x[0].reshape(28, 28), cmap='autumn')
        plt.show()


def parse_train_y_file():
    with open(train_y_file, 'rb') as file:
        desc_format = struct.Struct('>ii')
        magic, total = desc_format.unpack_from(file.read(desc_format.size))
        print(magic, total)

        data_format = struct.Struct('>%sB' % total)
        train_y = np.array(data_format.unpack_from(file.read(data_format.size)))
        np.save(np_train_y_file, train_y)


def parse_test_x_file():
    with open(test_x_file, 'rb') as file:
        desc_format = struct.Struct('>iiii')
        data_format = struct.Struct('>784B')

        # parse 16 byte
        magic, total, rows, cols = desc_format.unpack_from(file.read(desc_format.size))
        print(magic, total, rows, cols)

        # parse data
        mnist_list = []
        for i in range(total):
            # parse 784 byte
            read_data = file.read(data_format.size)
            data = data_format.unpack_from(read_data)
            mnist_list.append(data)

        train_x = np.array(mnist_list)
        np.save(np_test_x_file, train_x)

        # show first image, reshape 28*28
        fig = plt.figure()
        plt.imshow(train_x[0].reshape(28, 28), cmap='autumn')
        plt.show()


def parse_test_y_file():
    with open(text_y_file, 'rb') as file:
        desc_format = struct.Struct('>ii')
        magic, total = desc_format.unpack_from(file.read(desc_format.size))
        print(magic, total)

        data_format = struct.Struct('>%sB' % total)
        train_y = np.array(data_format.unpack_from(file.read(data_format.size)))
        np.save(np_test_y_file, train_y)


if __name__ == '__main__':
    parse_train_x_file()
    parse_train_y_file()
    parse_test_x_file()
    parse_test_y_file()
