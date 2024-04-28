import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
image_path = './data/Lena.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 이미지 크기 축소
resized_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

# DFT를 위한 함수 정의
def dft(img):
    # 이미지 데이터 타입을 float32로 변환
    img_float32 = np.float32(img)
    # 이미지 데이터를 FFT 수행
    f = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    # 결과의 크기 스펙트럼 계산
    magnitude_spectrum = 20 * np.log(cv2.magnitude(f[:, :, 0], f[:, :, 1]))
    return magnitude_spectrum

# 주파수 도메인으로 변환
dft_img = dft(resized_image)

# 입력 받은 반지름
r1 = int(input("첫 번째 원의 반지름 입력: "))
r2 = int(input("두 번째 원의 반지름 입력: "))

# 중심 좌표 및 크기 계산
rows, cols = resized_image.shape
center_row, center_col = rows // 2, cols // 2
x, y = np.ogrid[:rows, :cols]

# 첫 번째 원의 반지름을 기준으로 원 밖의 영역을 선택하는 마스크 생성
mask1 = np.sqrt((x - center_row)**2 + (y - center_col)**2) > r1

# 두 번째 원의 반지름을 기준으로 원 안의 영역을 선택하는 마스크 생성
mask2 = np.sqrt((x - center_row)**2 + (y - center_col)**2) < r2

# 원 밖의 영역을 선택하는 마스크와 원 안의 영역을 선택하는 마스크를 AND 연산하여 bandpass 필터 생성
bandpass_filter = np.logical_and(mask1, mask2)

# 필터링
dft_img_filtered = dft_img * bandpass_filter

# 역 DFT
f_ishift = np.fft.ifftshift(dft_img_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# 결과 출력
plt.subplot(121),plt.imshow(resized_image, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Band Pass Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()
