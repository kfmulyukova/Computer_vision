import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def to_halftone(img):
    halftone_img = np.empty(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            mean = np.mean(img[i][j])
            for k in range(img.shape[2]):
                halftone_img[i][j][k] = mean
    return halftone_img

def gauss(x, y, disp):
    return np.exp(-(x**2+y**2)/(2*(disp**2)))/(2*np.pi*(disp**2))

def gauss_K_table(K_size, disp):
    K_table = np.zeros((K_size, K_size), np.float64)
    pad = K_size//2
    for i in range(K_size):
        for j in range(K_size):
            K_table[i][j] = gauss(i - pad, j - pad, disp)
    K_table /= K_table.sum()
    return K_table

def gauss_filter(img, K_size):
    pad = K_size//2
    gauss_img = img.copy()
    gauss_table = gauss_K_table(K_size, 0.5)
    for i in range(pad, img.shape[0] - pad):
        for j in range(pad, img.shape[1] - pad):
            color = 0
            for g_i in range(K_size):
                for g_j in range(K_size):
                    color += img[i-g_i-pad][j-g_j-pad][0].astype(np.float64)*gauss_table[g_i][g_j]
            for k in range(img.shape[2]):
                gauss_img[i][j][k] = color
    return gauss_img

img = plt.imread("1.jpg")
plt.imshow(img)
plt.show()
img = to_halftone(img)
plt.imshow(img)
plt.show()
img = gauss_filter(img, 3)
plt.imshow(img)
plt.show()

gx = [-1, 0, 1,  -2, 0, 2,  -1, 0, 1]
gy = [1, 2, 1,    0, 0, 0,  -1, -2, -1]

def get_grad_magn(img):
    grad = np.zeros((img.shape[0], img.shape[1]))
    magn = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            arr = []
            for i1 in range(-1, 2):
                for j1 in range(-1, 2):
                    arr.append(img[i + i1][j + j1][0])
            g1 = np.dot(arr, gx)
            g2 = np.dot(arr, gy)
            grad[i][j] = (g1**2 + g2**2)**0.5
            if (g1 == 0):
                magn[i][j] = 0
            else:
                magn[i][j] = np.arctan(g2/g1)
    return grad, magn
grad, magn = get_grad_magn(img)
plt.imshow(grad, cmap='gray')
plt.show()

def angle_round(arr):
    res = np.zeros(arr.shape, int)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for angle in range(-360, 361, 45):
                if angle - 45//2 + 1 < arr[i][j] * 180 / np.pi < angle + 45//2 + 1: # для 90: если от 69 до 113, то 90
                    res[i][j] = angle
                    if res[i][j] < 0:
                        res[i][j] = 360 + res[i][j]
    return res
magn_rounded = angle_round(magn)

def suppression(grad, magn):
    for i in range(1, grad.shape[0] - 1):
        for j in range(1, grad.shape[1] - 1):
            if grad[i][j] != 0:
                if magn[i][j] == 0 or magn[i][j] == 180:
                    if not (grad[i][j] > grad[i][j - 1] and grad[i][j] > grad[i][j + 1]):
                        grad[i][j] = 0
                elif magn[i][j] == 45 or magn[i][j] == 225:
                    if not (grad[i][j] > grad[i - 1][j + 1] and grad[i][j] > grad[i + 1][j - 1]):
                        grad[i][j] = 0
                elif magn[i][j] == 90 or magn[i][j] == 270:
                    if not (grad[i][j] > grad[i - 1][j] and grad[i][j] > grad[i + 1][j]):
                        grad[i][j] = 0
                elif magn[i][j] == 135 or magn[i][j] == 315:
                    if not (grad[i][j] > grad[i - 1][j - 1] and grad[i][j] > grad[i + 1][j + 1]):
                        grad[i][j] = 0
suppression(grad, magn_rounded)
plt.imshow(grad, cmap='gray')
plt.show()

def segment(i, j, marks, grad, high, low):
    marks[i][j] = 1
    de = deque()
    for i1 in range(-1, 2):
        for j1 in range(-1, 2):
            if marks[i + i1][j + j1] == 0 and grad[i + i1][j + j1] > low:
                de.append([i + i1, j + j1])
                marks[i + i1][j + j1] = 1
    while (de):
        cur = de.pop()
        if marks[i][j] == 0 and grad[i][j] > high and \
                cur[0] < grad.shape[0] - 1 and cur[1] < grad.shape[1] - 1 and \
                cur[0] > 0 and cur[1] > 0:
            for i1 in range(-1, 2):
                for j1 in range(-1, 2):
                    if marks[cur[0] + i1][cur[1] + j1] == 0 and grad[cur[0] + i1][cur[1] + j1] > low:
                        de.append([cur[0] + i1, cur[1] + j1])
                        marks[cur[0] + i1][cur[1] + j1] = 1

def hysteresis(grad, high, low):
    marks = np.zeros((grad.shape[0], grad.shape[1]), np.uint8)
    for i in range(1, grad.shape[0] - 1):
        for j in range(1, grad.shape[1] - 1):
            if marks[i][j] == 0 and grad[i][j] > high:
                segment(i, j, marks, grad, high, low)

    res = np.zeros(grad.shape, np.uint8)
    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            if marks[i][j] == 1:
                res[i][j] = 255
    return res

plt.imshow(hysteresis(grad, 200, 100), cmap='gray')
plt.show()


img = plt.imread("2.jpg")
plt.imshow(img)
plt.show()
img = to_halftone(img)
plt.imshow(img)
plt.show()
img = gauss_filter(img, 3)
plt.imshow(img)
plt.show()
grad, magn = get_grad_magn(img)
plt.imshow(grad, cmap='gray')
plt.show()
magn_rounded = angle_round(magn)
suppression(grad, magn_rounded)
plt.imshow(grad, cmap='gray')
plt.show()
plt.imshow(hysteresis(grad, 200, 100), cmap='gray')
plt.show()


img = plt.imread("3.jpg")
plt.imshow(img)
plt.show()
img = to_halftone(img)
plt.imshow(img)
plt.show()
img = gauss_filter(img, 3)
plt.imshow(img)
plt.show()
grad, magn = get_grad_magn(img)
plt.imshow(grad, cmap='gray')
plt.show()
magn_rounded = angle_round(magn)
suppression(grad, magn_rounded)
plt.imshow(grad, cmap='gray')
plt.show()
plt.imshow(hysteresis(grad, 200, 100), cmap='gray')
plt.show()
