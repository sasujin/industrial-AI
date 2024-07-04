import cv2
import numpy as np
import matplotlib.pyplot as plt

# 히스토그램 평탄화 함수
def histogram_equalization(image):
    # Check if image is color (3 channels) or grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image  # Assuming input is already grayscale

    # Equalize histogram
    equalized_image = cv2.equalizeHist(gray_image)

    # Calculate histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Normalize histogram
    hist_normalized = hist.ravel() / hist.max()

    return equalized_image, hist_normalized

# HSV 컬러 스페이스에서 V 채널에 대한 히스토그램 평탄화 함수
def hsv_value_equalization(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    equalized_v = cv2.equalizeHist(v)
    equalized_hsv_image = cv2.merge([h, s, equalized_v])
    equalized_color_image = cv2.cvtColor(equalized_hsv_image, cv2.COLOR_HSV2BGR)
    hist = cv2.calcHist([v], [0], None, [256], [0, 256])
    hist_normalized = hist.ravel() / hist.max()
    return equalized_color_image, hist_normalized

# 이미지 불러오기
input_image_path = './data/Lena.png'
input_image = cv2.imread(input_image_path)

# 이미지의 채널 수가 3개인지 확인
if input_image is None or input_image.shape[2] != 3:
    print("Error: Unable to load image or input image does not have 3 channels (BGR).")
    exit()

# 사용자로부터 채널을 선택하도록 입력
channel = input("Enter the channel to perform histogram equalization (R/G/B): ").upper()

# 사용자 입력에 따라 히스토그램 평활화를 수행합
if channel in ('R', 'G', 'B'):
    channel_index = {'R': 2, 'G': 1, 'B': 0}[channel]
    equalized_image, hist_normalized = histogram_equalization(input_image[:, :, channel_index])
    # Display histogram
    plt.plot(hist_normalized, color='gray')
    plt.xlabel('Intensity')
    plt.ylabel('Normalized Frequency')
    plt.title('Histogram')
    plt.show()
    # Display histogram equalized image
    cv2.imshow("Histogram Equalized Image", equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif channel == 'V':
    hsv_equalized_image, hist_normalized_hsv = hsv_value_equalization(input_image)
    # Display histogram
    plt.plot(hist_normalized_hsv, color='gray')
    plt.xlabel('Intensity')
    plt.ylabel('Normalized Frequency')
    plt.title('HSV Value Histogram')
    plt.show()
    # Display HSV value equalized image
    cv2.imshow("HSV Value Equalized Image", hsv_equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Invalid channel selection.")
