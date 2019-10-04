import numpy as np
import os
import skimage
from joblib import Parallel, delayed

ud = []
lr = []

def handle_image(i, ud, lr, images, m, up, down, left, right):
    if i % 50 == 0:
        print(i / len(images) * 100)
    udi = np.zeros((m, m, m, m), dtype=np.uint16)
    lri = np.zeros((m, m, m, m), dtype=np.uint16)
    for x1 in range(m):
        for y1 in range(m):
            for x2 in range(m):
                for y2 in range(m):
                    udi[x1][y1][x2][y2] = np.abs(up[i][x1][y1] - down[i][x2][y2]).sum()
                    lri[x1][y1][x2][y2] = np.abs(left[i][x1][y1] - right[i][x2][y2]).sum()
    ud[i] = udi
    lr[i] = lri

def dist(path, data_path, p, prefix):
    m = 512 // p
    files = sorted(os.listdir(path))
    images = []
    for file in files:
        full_path = os.path.join(path, file)
        image = skimage.io.imread(full_path)
        images.append(image)
    left = np.zeros((len(images), m, m, p, 3), dtype=np.uint8)
    right = np.zeros((len(images), m, m, p, 3), dtype=np.uint8)
    up = np.zeros((len(images), m, m, p, 3), dtype=np.uint8)
    down = np.zeros((len(images), m, m, p, 3), dtype=np.uint8)
    for i in range(len(images)):
        image = images[i]
        if len(image.shape) == 2:
            image = np.stack((image // 256,)*3, axis=-1)
        #print(i, image.max())
        for x in range(m):
            for y in range(m):
                left[i][x][y] = image[p * x, p * y:p * (y + 1)]
                right[i][x][y] = image[p * x + p - 1, p * y:p * (y + 1)]
                up[i][x][y] = image[p * x: p * (x + 1), p * y]
                down[i][x][y] = image[p * x: p * (x + 1), p * y + p - 1]
    ud = np.zeros((len(images), m, m, m, m), dtype=np.uint16)
    lr = np.zeros((len(images), m, m, m, m), dtype=np.uint16)

    Parallel(n_jobs=10, require='sharedmem')(delayed(handle_image)(i, ud, lr, images, m, up, down, left, right) for i in range(len(images)))
    np.save(os.path.join(data_path, 'ud' + prefix), ud)
    np.save(os.path.join(data_path, 'lr' + prefix), lr)

dist('C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\64-sources',\
     'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\data',
     64, '64-sources')
dist('C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\64',\
     'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\data',
     64, '64')
dist('C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\32-sources',\
     'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\data',
     32, '32-sources')
dist('C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\32',\
     'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\data',
     32, '32')
dist('C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\16',\
     'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\data',
     16, '16')
dist('C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\16-sources',\
     'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\data',
     16, '16-sources')
