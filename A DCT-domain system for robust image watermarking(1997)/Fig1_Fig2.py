import cv2
import numpy as np
import matplotlib.pyplot as plt
from watermarking_function import detect_watermark, water_marking

M = 16000
alpha = 0.2

# watermarking
image = cv2.imread('./boat.png', cv2.IMREAD_GRAYSCALE)
res, x= water_marking(image,M, alpha)

org_image = cv2.imread('./boat.png', cv2.IMREAD_GRAYSCALE)
fig = plt.figure(figsize = (15,15))
ax1 = fig.add_subplot(1,2,1)
ax1.set_title("original image")
plt.imshow(org_image, cmap = 'gray')

ax2 = fig.add_subplot(1,2,2)
ax2.set_title("watermarked image")
plt.imshow(res, cmap = 'gray')
plt.show()

# detect watermark
y = np.random.normal(0, 1, size=(1000, len(x)))
y[100] = x
detect, threshold = detect_watermark(res, y, M, alpha)
plt.plot(detect)
plt.axhline(y=threshold, color = 'red')
plt.show()




