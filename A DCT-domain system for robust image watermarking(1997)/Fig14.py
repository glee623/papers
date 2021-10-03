import cv2
import numpy as np
import matplotlib.pyplot as plt
from watermarking_function import water_marking, detect_watermark

M = 16000
alpha = 0.2

# watermarking
image = cv2.imread('./boat.png', cv2.IMREAD_GRAYSCALE)

res1, x1= water_marking(image,M, alpha)

res2, x2= water_marking(res1,M, alpha)

res3, x3= water_marking(res2,M, alpha)

res4, x4= water_marking(res3,M, alpha)

res5, x5= water_marking(res4,M, alpha)

y = np.random.normal(0, 1, size=(1000, len(x1)))
y[100] = x1
y[300] = x2
y[500] = x3
y[700] = x4
y[900] = x5

detect, threshold = detect_watermark(res5, y, M, alpha)
plt.imshow(res5, cmap='gray')
plt.show()
plt.plot(detect)
plt.axhline(y=threshold, color = 'red')
plt.show()


