import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import deque
import random

img = plt.imread("1.jpg")
plt.imshow(img)

def to_halftone(img):
    halftone_img = np.empty(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            mean = np.mean(img[i][j])
            for k in range(img.shape[2]):
                halftone_img[i][j][k] = mean
    return halftone_img

halftone_img = to_halftone(img)
plt.imshow(halftone_img)
plt.show()

def otsu(img):
    y = np.zeros(256)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            y[img[i, j]] += 1
    size = img.shape[0] * img.shape[1]
    P = y/size
    u = 0
    t = 1
    T = u
    Smin = sys.maxsize
    for u in range(0, 256, t):
        q1 = 0
        m1 = 0
        m2 = 0
        sigma1 = 0
        sigma2 = 0
        for i in range(0, u + 1):
            q1 += P[i - 1]
        q2 = 1 - q1
        for i in range(0, u + 1):
            m1 += i * P[i]
        if q1 == 0 or q2 == 0:
            continue
        m1 /= q1
        for i in range(u + 1, 256):
            m2 += i * P[i]
        m2 /= q2
        for i in range(0, u + 1):
            sigma1 += (i - m1)**2 * P[i]
        sigma1 /= q1
        for i in range(u + 1, 256):
            sigma2 += (i - m2)**2 * P[i]
        sigma2 /= q2
        sigma = q1*sigma1 + q2*sigma2
        if sigma < Smin:
            Smin = sigma
            T = u
    return T

T = otsu(halftone_img)

def to_black_and_white(img, T):
    bw_img = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if img[i][j][k] > T:
                    bw_img[i][j] = [255, 255, 255]
    return bw_img

bw_img = to_black_and_white(halftone_img, T)
plt.imshow(bw_img)
plt.imsave("black_and_white.jpg", bw_img)
plt.show()

def salt_and_pepper(img):
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            diff = 0
            for i1 in range(-1, 2):
                for j1 in range(-1, 2):
                    diff += img[i + i1][j + j1][0]
            if diff == 2040:
                img[i][j] = [255, 255, 255]
            if diff == 255:
                img[i][j] = [0, 0, 0]

salt_and_pepper(bw_img)
plt.imshow(bw_img, cmap='gray')
plt.imsave("salt_and_pepper.jpg", bw_img)
plt.show()

def segment(i, j, marks, img, cur_seg):
    marks[i][j] = cur_seg
    de = deque()
    for i1 in range(-1, 2):
        for j1 in range(-1, 2):
            if marks[i + i1][j + j1] == 0:
                de.append([i + i1, j + j1])
                marks[i + i1][j + j1] = cur_seg
    while (de):
        cur = de.pop()
        if img[cur[0]][cur[1]][0] == img[i][j][0] and cur[0] < img.shape[0] - 1 and cur[1] < img.shape[1] - 1 and \
                cur[0] > 0 and cur[1] > 0:
            for i1 in range(-1, 2):
                for j1 in range(-1, 2):
                    if marks[cur[0] + i1][cur[1] + j1] == 0:
                        de.append([cur[0] + i1, cur[1] + j1])
                        marks[cur[0] + i1][cur[1] + j1] = cur_seg

def segment_bw(img):
    marks = np.zeros((img.shape[0], img.shape[1]), np.uint32)
    cur_seg = 0
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if marks[i][j] == 0:
                cur_seg += 1
                segment(i, j, marks, img, cur_seg)
    colors = []
    for i in range(cur_seg):
        colors.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
    seg_img = np.zeros(img.shape, np.uint8)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            seg_img[i][j] = colors[marks[i][j] - 1] #для 1-го сегмента 0вой цвет, для 2го первый и тд,
                                                    # тк marks заполняется количеством сегментов
    return seg_img

seg_img = segment_bw(bw_img)
plt.imshow(seg_img)
plt.imsave("seg.jpg", seg_img)
plt.show()

img = plt.imread("2.jpg")
plt.imshow(img)
plt.show()

halftone_img = to_halftone(img)
bw_img = to_black_and_white(halftone_img, otsu(halftone_img))
plt.imshow(bw_img)
plt.show()

salt_and_pepper(bw_img)
plt.imshow(segment_bw(bw_img))
plt.show()

img = plt.imread("3.jpg")
plt.imshow(img)
plt.show()

halftone_img = to_halftone(img)
bw_img = to_black_and_white(halftone_img, otsu(halftone_img))
plt.imshow(bw_img)
plt.show()
salt_and_pepper(bw_img)
plt.imshow(segment_bw(bw_img))
plt.show()

gx = [-1, 0, 1,  -2, 0, 2,  -1, 0, 1]
gy = [1, 2, 1,    0, 0, 0,  -1, -2, -1]
img = plt.imread("1.jpg")
img = to_halftone(img)
grad = np.zeros((img.shape[0], img.shape[1]))
for i in range(1, img.shape[0] - 1):
    for j in range(1, img.shape[1] - 1):
        arr = []
        for i1 in range(-1, 2):
            for j1 in range(-1, 2):
                arr.append(img[i + i1][j + j1][0])
        g1 = np.dot(arr, gx) / 2
        g2 = np.dot(arr, gy) / 2
        grad[i][j] = (g1**2 + g2**2)**0.5

plt.imshow(grad, cmap='gray')
plt.show()

img = plt.imread("1.jpg")
img = to_halftone(img)

y = np.zeros(256)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        y[img[i, j]] += 1
plt.figure()
plt.bar([i for i in range(256)], y)
plt.show()

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j][0] < 150:
            img[i][j] = [0, 0, 0]
        else:
            img[i][j] = [255, 255, 255]
# salt_and_pepper(img)
plt.imshow(img)
plt.show()

plt.imshow(segment_bw(img))
plt.show()

img = plt.imread("2.jpg")
img = to_halftone(img)
plt.imshow(img)
plt.show()
y = np.zeros(256)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        y[img[i, j]] += 1
plt.figure()
plt.bar([i for i in range(256)], y)
plt.show()

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j][0] < 150:
            img[i][j] = [0, 0, 0]
        else:
            img[i][j] = [255, 255, 255]
salt_and_pepper(img)
plt.imshow(img)
plt.show()
plt.imshow(segment_bw(img))
plt.show()

img = plt.imread("3.jpg")
img = to_halftone(img)
plt.imshow(img)
plt.show()
y = np.zeros(256)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        y[img[i, j]] += 1
plt.figure()
plt.bar([i for i in range(256)], y)
plt.show()

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j][0] < 80:
            img[i][j] = [0, 0, 0]
        else:
            img[i][j] = [255, 255, 255]
salt_and_pepper(img)
plt.imshow(img)
plt.show()
plt.imshow(segment_bw(img))
plt.show()
