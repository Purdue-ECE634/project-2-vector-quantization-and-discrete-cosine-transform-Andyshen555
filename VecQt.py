import cv2
import numpy as np
import math
import os

image_path = './img/goldhill.png'
folder = './img/'
lv = 4

def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def psnr(img1, img2):
    result = mse(img1, img2)
    if result == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(result))

def vq(lut, image):
	H, W = np.shape(image)
	output = np.zeros((H, W))

	for i in range(H//4):
		for j in range(W//4):
			patch = image[4*i : 4*(i+1), 4*j : 4*(j+1)]
			mse_max = np.inf
			for k in range(lut.shape[0]):
				mse_value = mse(patch, lut[k])
				if mse_value < mse_max:
					lv = k
					mse_max = mse_value
			output[4*i : 4*(i+1), 4*j : 4*(j+1)] = lut[lv]
	return output

train_list = []
# for image in os.listdir(folder):
#   img = cv2.imread(folder+image)
#   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   H, W = np.shape(img)
#   for i in range(H//4):
#     for j in range(W//4):
#       train_img = img[4*i : 4*(i+1), 4*j : 4*(j+1)]
#       train_list.append(train_img)
# train_num = 10

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = np.shape(img)
for i in range(H//4):
  for j in range(W//4):
    train_img = img[4*i : 4*(i+1), 4*j : 4*(j+1)]
    train_list.append(train_img)

train_num = len(train_list)
lut = np.arange(0, 256, 256 // lv)

lut = np.tile(lut.reshape((lv, 1)), 16)
lut = lut.reshape((lv, 4, 4))
train_list = np.asarray(train_list)
itr = 0

prev_error = 1

while(True):
  vector = np.zeros((train_num))
  errors = np.zeros((train_num))

  for i in range(train_num):
    min_MSE = np.inf
    for j in range(lv):
      mse_value = mse(train_list[i], lut[j])
      if min_MSE > mse_value:
        min_MSE = mse_value
        quantizor = j
    vector[i] = quantizor
    errors[i] = min_MSE
  error = np.mean(errors)

  if (np.abs(error - prev_error) / prev_error) <= 0.02:
    break
  else:
    for l in range(lv):
      if np.sum(vector == l) != 0: 
        lut[l] = np.mean(train_list[vector == l], axis=0)
      else:
        lut[l] = np.zeros((4, 4))
    prev_error = error

  itr += 1

output = vq(lut, img)
  
PSNR = psnr(img, output)
print(PSNR)

cv2.imshow("img", img)
cv2.imshow("output", output.astype(np.uint8))
cv2.waitKey(0)

