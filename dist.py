import numpy as np
import os
import skimage
#from joblib import Parallel, delayed
from datetime import datetime
from numba import jit
from multiprocessing import Process

def dist(path, data_path, p, prefix):
    print('started', prefix)
    start = datetime.now()
    m = 512 // p
    files = sorted(os.listdir(path))
    images = []
    for file in files:
        full_path = os.path.join(path, file)
        image = skimage.io.imread(full_path)
        images.append(image)
    left = np.zeros((len(images), m, m, p, 3), dtype=np.int32)
    right = np.zeros((len(images), m, m, p, 3), dtype=np.int32)
    up = np.zeros((len(images), m, m, p, 3), dtype=np.int32)
    down = np.zeros((len(images), m, m, p, 3), dtype=np.int32)
    for i in range(len(images)):
        image = images[i]
        if len(image.shape) == 2:
            image = np.stack((image // 256,)*3, axis=-1)
        for x in range(m):
            for y in range(m):
                left[i][x][y] = image[p * x, p * y:p * (y + 1)]
                right[i][x][y] = image[p * x + p - 1, p * y:p * (y + 1)]
                up[i][x][y] = image[p * x: p * (x + 1), p * y]
                down[i][x][y] = image[p * x: p * (x + 1), p * y + p - 1]
    ud = np.zeros((len(images), m, m, m, m), dtype=np.uint16)
    lr = np.zeros((len(images), m, m, m, m), dtype=np.uint16)
    compute(m, p, left, right, up, down, len(images), ud, lr)
    np.save(os.path.join(data_path, 'ud' + prefix), ud)
    np.save(os.path.join(data_path, 'lr' + prefix), lr)
    print(prefix, datetime.now() - start)

@jit(nopython=True)
def compute(m, p, left, right, up, down, image_len, ud, lr):
    for i in range(image_len):
        #if i % 50 == 0:
        #    print(i / image_len * 100)
        for x1 in range(m):
            for y1 in range(m):
                for x2 in range(m):
                    for y2 in range(m):
                        ud[i][x1][y1][x2][y2] = np.abs(up[i][x1][y1] - down[i][x2][y2]).sum()
                        lr[i][x1][y1][x2][y2] = np.abs(left[i][x1][y1] - right[i][x2][y2]).sum()


arg1 = ['C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\64-sources',\
       'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\64',\
       'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\32-sources',\
       'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\32',\
       'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\16-sources',\
       'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\16',\
       'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_test1_blank\\64',\
       'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_test1_blank\\32',\
       'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_test1_blank\\16',\
       'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_test2_blank\\64',\
       'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_test2_blank\\32',\
       'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_test2_blank\\16']

arg2 = 'C:\\Users\\agano\\Documents\\notebooks\\huawei\\data_train\\data'
arg3 = [64, 64, 32, 32, 16, 16, 64, 32, 16, 64, 32, 16]
arg4 = ['train-64-sources', 'train-64', 'train-32-sources', 'train-32', 'train-16-sources', 'train-16',\
        'test1-64', 'test1-32', 'test1-16', 'test2-64', 'test2-32', 'test2-16']
#Parallel(n_jobs=10, prefer="threads")(delayed(dist)(arg1[i], arg2, arg3[i], arg4[i]) for i in range(len(arg3)))
#for i in range(len(arg3)):
#    dist(arg1[i], arg2, arg3[i], arg4[i])

if __name__ == '__main__':
    p = []
    for i in range(len(arg3)):
        p.append(Process(target=dist, args=(arg1[i], arg2, arg3[i], arg4[i])))
        p[-1].start()
    for i in range(len(p)):
        p[i].join()
