import cv2
import numpy as np
import matplotlib.pyplot as plt
from watermarking_function import water_marking, detect_watermark

M = 16000
alpha = 0.2

# watermarking
image = cv2.imread('./boat.png', cv2.IMREAD_GRAYSCALE)
res, x= water_marking(image,M, alpha)

#low pass filter
kernel = np.ones((5,5), np.float32) / 25
lowpass_image = cv2.filter2D(res, -1, kernel)

#detect watermark
y = np.random.normal(0, 1, size=(1000, len(x)))
y[100] = x
detect, threshold = detect_watermark(lowpass_image, y, M, alpha)

plt.imshow(lowpass_image, cmap='gray')
plt.show()

plt.plot(detect)
plt.axhline(y=threshold, color = 'red')
plt.show()


