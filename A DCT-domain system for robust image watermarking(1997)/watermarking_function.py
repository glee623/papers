import cv2
import numpy as np

def water_marking(org_image, M, alpha):
    # 0~1 사이 float형 image로 변경하여 dct 변환
    image = np.float32(org_image) / 255.0
    dct_image = cv2.dct(image)

    #0~255 범위 zig-zag 스캔
    zig_zag = []
    for i in range(255):
        for j in range(i + 1):
            if i % 2 == 0:
                zig_zag.append(dct_image[i - j, j])
            else:
                zig_zag.append(dct_image[j, i - j])
    zig_zag = np.asarray(zig_zag)
    # zig_zag_picked : watermarking 할 M ~ 2M 범위 slicing
    zig_zag_picked = zig_zag[M : 2 * M]

    # normal distribution 랜덤 변수 x 생성 (watermark)
    x = np.random.normal(0, 1, size=M)

    # zig_zag_picked에 watermark x 적용
    for i in range(len(zig_zag_picked)):
        zig_zag_picked[i] += (alpha * np.abs(zig_zag_picked[i]) * x[i])
    zig_zag[M : 2 * M] = zig_zag_picked

    # watermarking 된 값 다시 zig_zag input
    index = 0
    for i in range(255):
        for j in range(i + 1):
            if i % 2 == 0:
                dct_image[i - j, j] = zig_zag[index]
            else:
                dct_image[j, i - j] = zig_zag[index]
            index += 1
    # inverse dct 하여 다시 image로 변환
    inv_dct = cv2.idct(dct_image)

    return inv_dct, x


def detect_watermark(image, y, M, alpha):
    # 0~1 사이 float형 image로 변경하여 dct 변환
    image = np.float32(image) / 255.0
    dct_image = cv2.dct(image)

    #0~255 범위 zig-zag 스캔
    zig_zag = []
    for i in range(255):
        for j in range(i + 1):
            if i % 2 == 0:
                zig_zag.append(dct_image[i - j, j])
            else:
                zig_zag.append(dct_image[j, i - j])
    #zig-zag 스캔한 0~255 범위 중 M~2M 범위 slicing
    zig_zag = np.asarray(zig_zag)[M : 2 * M]

    # zig_zag의 평균 값과 alpha값 이용하여 watermark 검출 기준 threshold 설정
    threshold = (alpha / 2) * np.mean(np.abs(zig_zag))

    # detect list에 np.mean(y[i] * zig_zag)를 append
    # - watermarked image를 지그재그 스캔한 zig_zag list와,
    # 길이가 M인 정규분포 난수 1000개를 각각 곱해 평균낸 값
    # 총 1000개를 detect list에 담는다
    detect = []
    for i in range(len(y)):
        detect.append(np.mean(y[i]*zig_zag))

    return detect, threshold