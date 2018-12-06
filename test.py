import numpy as np
import os
from PIL import Image
from sklearn.externals import joblib

lr_clf = joblib.load('./model/lr.model')
rf_clf = joblib.load('./model/rf.model')
ada_boost_clf = joblib.load('./model/ada_rf.model')


def predict(file_path):
    """
    图片笔画需要粗
    先把图片转成灰度模式
    按照MNIST数据格式，处理每张图片，每个像素为0无内容，1-255为有内容，数值表示灰度值
    """
    file_name = os.path.basename(file_path)
    print(file_name)
    im = Image.open(file_path)
    im = im.convert('L')
    im = im.resize((28, 28))

    # # numpy可直接从灰度模式Image对象获取数组
    # im = im.convert('L')
    # im = im.resize((28, 28))
    # x = np.array(im)

    # im.getdata()也可以获取到像素数组
    # im = im.convert('L')
    # color_list = [255 - item for item in im.getdata()]
    # img = Image.fromarray(np.array(color_list).reshape(28, 28), 'RGBA')
    # img.show()

    color_list = []
    w, h = im.size
    for i in range(w):
        for j in range(h):
            temp = 1.0 - float(im.getpixel((j, i))) / 255.0
            color_list.append(temp)
            im.putpixel((j, i), int(temp * 255))
    im.show()

    x = np.array(color_list)
    x = x.reshape(-1, 28 * 28)

    print('lr_clf', lr_clf.predict(x))
    print('rf_clf', rf_clf.predict(x))
    print('ada_boost_clf', ada_boost_clf.predict(x))
    print('-----------------')


def main(path):
    for name in os.listdir(path):
        file_path = os.path.join(path, name)
        predict((file_path))


if __name__ == '__main__':
    file_path = './test'
    main(file_path)