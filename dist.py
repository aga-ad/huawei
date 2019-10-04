import numpy as np
import os
import skimage
from joblib import Parallel, delayed

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

    for i in range(len(images)):
        if i % 50 == 0:
            print(i / len(images) * 100)
        for x1 in range(m):
            for y1 in range(m):
                for x2 in range(m):
                    for y2 in range(m):
                        ud[i][x1][y1][x2][y2] = np.abs(up[i][x1][y1] - down[i][x2][y2]).sum()
                        lr[i][x1][y1][x2][y2] = np.abs(left[i][x1][y1] - right[i][x2][y2]).sum()
    np.save(os.path.join(data_path, 'ud' + prefix), ud)
    np.save(os.path.join(data_path, 'lr' + prefix), lr)


arg1 = ['C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\64-sources',\
       'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\64',\
       'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\32-sources',\
       'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\32',\
       'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\16-sources',\
       'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\16']
arg2 = 'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\data'
arg3 = [64, 64, 32, 32, 16, 16]
arg4 = ['64', '64', '32', '32', '16', '16']
Parallel(n_jobs=6)(delayed(dist)(arg1[i], arg2, arg3[i], arg4[i]) for i in range(6))
