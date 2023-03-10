import cv2
import numpy as np
import math
from scipy.fftpack import dct, idct
from enum import Enum

class DCTtable(Enum):
    right = 0
    lowerL = 1
    down = 2
    upperR = 3

image_path = './img/goldhill.png'
K = 32
 
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def DCT(patch):
  dct_col = dct(patch, axis=0, norm='ortho')
  return dct(dct_col, axis=1, norm='ortho')

def IDCT(patch):
  idct_col = idct(patch, axis=0, norm='ortho')
  return idct(idct_col, axis=1, norm='ortho')

def compute_coeff(patch, K):
  H, W = patch.shape
  result = np.zeros_like(patch)	
  i = 0
  j = 0
  next = DCTtable.right

  for k in range(K):
    result[i, j] = patch[i, j]
    if next == DCTtable.right:
      if k == 0: 
        j += 1
        next = DCTtable.upperR
      else: 
        i +=1
        j -= 1
        next = DCTtable.upperR

    elif next == DCTtable.upperR:
      if j == 0:
        i += 1
        next = DCTtable.lowerL
      else:
        i += 1
        j -= 1
        next = DCTtable.upperR	

    elif next == DCTtable.down:
      if i == 0:
        j +=1
        next = DCTtable.right
      elif j == (W-1):
        i += 1
        next = DCTtable.lowerL
      else:
        i -= 1
        j += 1
        next = DCTtable.down
  
    elif next == DCTtable.lowerL:
      if i == (H-1): 
        j += 1
        next = DCTtable.right
      elif j == (W-1):
        i += 1
        j -= 1
        next = DCTtable.upperR
      else: 
        i -= 1
        j += 1
        next = DCTtable.down

  return result

img = cv2.imread(image_path, 0)
# cv2.imshow("img", img)
output = np.zeros_like(img)
H, W = np.shape(img)

for i in range(H//8):
  for j in range(W//8):
    patch = img[8*i : 8*(i+1), 8*j : 8*(j+1)]
    dct_block = DCT(patch)
    temp_block = compute_coeff(dct_block, K)
    output[8*i : 8*(i+1), 8*j : 8*(j+1)] = IDCT(temp_block)


PSNR = psnr(img, output)
print(PSNR)
cv2.imwrite("output.png", output.astype(np.uint8))
# cv2.waitKey(0)

