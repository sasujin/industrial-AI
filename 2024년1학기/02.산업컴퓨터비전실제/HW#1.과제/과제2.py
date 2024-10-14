import cv2
import numpy as np

# 이미지 불러오기
image = cv2.imread('./data/Lena.png', cv2.IMREAD_GRAYSCALE)

# 임의의 노이즈 생성
noise = np.random.normal(loc=0, scale=150, size=image.shape).astype(np.uint8)
noisy_image = cv2.add(image, noise)

# 필터링
gaussian_filtered = cv2.GaussianBlur(noisy_image, (5, 5), 0)
median_filtered = cv2.medianBlur(noisy_image, 5)
bilateral_filtered = cv2.bilateralFilter(noisy_image, 9, 75, 75)

# 결과 출력
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow('Gaussian Filtered', gaussian_filtered)
cv2.imshow('Median Filtered', median_filtered)
cv2.imshow('Bilateral Filtered', bilateral_filtered)

# 입력 영상과의 차이 계산 및 출력
cv2.imshow('Difference Gaussian', np.abs(image - gaussian_filtered))
cv2.imshow('Difference Median', np.abs(image - median_filtered))
cv2.imshow('Difference Bilateral', np.abs(image - bilateral_filtered))

cv2.waitKey(0)
cv2.destroyAllWindows()
