import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('digits.png', 0)
cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]
x = np.array(cells)


img_test = cv2.imread('so3.png', 0)
x2 = np.array(img_test)
train = x[:, :50].reshape(-1, 400).astype(np.float32)

test2 = x2.reshape(-1, 400).astype(np.float32)

k = np.arange(10)

train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = train_labels.copy()
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

kq1, kq2, kq3, kq4 = knn.findNearest(test2, k=5)
print(kq1)
print(kq2)
print(kq3)
print(kq4)