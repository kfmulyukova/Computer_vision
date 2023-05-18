import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = np.array(Image.open("Cats.jpg"))

#инвертирование
inverted_img = Image.fromarray(255 - img)
plt.imshow(inverted_img)
plt.show()

#полутоновое
halftone_img = np.empty(img.shape, dtype=np.uint8)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        halftone_img[i][j] = np.mean(img[i][j])
plt.imshow(halftone_img)
plt.show()

# добавление случайного шума
make_some_noise_img = np.empty(img.shape, dtype=np.uint8)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(img.shape[2]):
            if np.random.random() < 0.5:
                make_some_noise_img[i, j] = halftone_img[i, j] + np.random.normal(0, 5)
            else:
                make_some_noise_img[i, j] = halftone_img[i, j]
plt.imshow(make_some_noise_img, cmap='gray')
plt.show()

# построение гистограммы
# полутонового изображения
x = [i for i in range(256)]
y_halftone = np.zeros(256)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        y_halftone[halftone_img[i, j]] += 1
plt.figure()
plt.bar(x, y_halftone)
plt.show()

#изображения с шумом
x = [i for i in range(256)]
y_noise = np.zeros(256)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        y_noise[make_some_noise_img[i, j]] += 1
plt.figure()
plt.bar(x, y_noise)
plt.show()

#размытие
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

K_size = 7
pad = K_size//2
gauss_img = make_some_noise_img.copy()
gauss_table = gauss_K_table(K_size, 3)
for i in range(pad, img.shape[0] - pad):
    for j in range(pad, img.shape[1] - pad):
        color = 0
        for g_i in range(K_size):
            for g_j in range(K_size):
                color += make_some_noise_img[i-g_i-pad][j-g_j-pad][0].astype(np.float64)*gauss_table[g_i][g_j]
        for k in range(img.shape[2]):
            gauss_img[i][j][k] = color
plt.imshow(gauss_img)
plt.show()

threshold = 20

mask_img = make_some_noise_img.copy()
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if abs(make_some_noise_img[i][j][0].astype(np.float64) - gauss_img[i][j][0]) > threshold:
            for k in range(img.shape[2]):
                if mask_img[i][j][k] + (255 - gauss_img[i][j][k]) * 0.15 > 255:
                    mask_img[i][j][k] = 255
                else:
                    mask_img[i][j][k] += (255 - gauss_img[i][j][k]) * 0.15
plt.imshow(mask_img)
plt.show()

y_mask = np.zeros(256)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        y_mask[mask_img[i, j]] += 1
plt.figure()
plt.bar(x, y_mask)
plt.show()

#выравнивание гистограммы, нормировка яркости
N = img.shape[0]*img.shape[1]
t = np.zeros(256, np.float64)
t[0] = y_mask[0]
for i in range(1, 256):
    t[i] = y_mask[i] + t[i - 1]
t_min = min(t)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        color = round((t[mask_img[i][j][0]] - t_min)/(N - t_min)*255.0)
        for k in range(img.shape[2]):
            mask_img[i][j][k] = color
plt.imshow(mask_img)
plt.show()

y_mask = np.zeros(256)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        y_mask[mask_img[i, j]] += 1
plt.figure()
plt.bar(x, y_mask)
plt.show()
