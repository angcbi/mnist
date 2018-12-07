
目录说明
-----

>>>>
- source_mnist/  原始MNIST数据集
- data/          numpy解析后的数据集
- model/         训练的模型
- test/          测试图片


解析MNIST数据集
------------

详情参考 [mnist数据集说明](http://yann.lecun.com/exdb/mnist/)

1. MNIST共6000张训练图片，1000张测试图片，共有四个二进制文件
>
- train-images-idx3-ubyte：  training set images 
- train-labels-idx1-ubyte:   training set labels 
- t10k-images-idx3-ubyte:    test set images 
- t10k-labels-idx1-ubyte:    test set labels

2. MNIST 文件格式说明

TRAINING SET LABEL FILE (train-labels-idx1-ubyte)

|[offset]| [type]|          [value]|          [description]|
|----|----|----|----| 
|0000|     32 bit integer|  0x00000801(2049)| magic number (MSB first) |
|0004|     32 bit integer|  60000|            number of items |
|0008|     unsigned byte||   ??  |            label |
|0009|     unsigned byte|   ??   |             label |
|xxxx|     unsigned byte|   ??|               label|

TRAINING SET IMAGE FILE (train-images-idx3-ubyte)

|[offset]| [type]|          [value]     |     [description] 
|----|----|----|----| 
|0000|     32 bit integer|  0x00000803(2051)| magic number |
|0004|     32 bit integer|  60000          | number of images |
|0008|     32 bit integer|  28             | number of rows |
|0012|     32 bit integer|  28             | number of columns |
|0016|     unsigned byte |  ??             | pixel |
|0017|     unsigned byte |  ??             | pixel |
|xxxx|     unsigned byte |  ??             | pixel|

Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  10000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 
........ 
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  10000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
........ 
xxxx     unsigned byte   ??               pixel

3 MNIST 图片格式说明
MNIST是28*28像素的灰度图片(对应PIL类型为L), 实际是一个二维矩阵，0表示无内容，1-255表示有内容，数值表示对应灰度值
