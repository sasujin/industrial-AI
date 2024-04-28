import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# #24.04.12 Test
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ['TF_GPU_ALLOCATOR'] = "cuda_malloc_async"
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "True"
os.environ['TF_CUDNN_RESET_RND_GEN_STATE'] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
'''
'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4*1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
'''

print("Tensorflow version : " + tf.__version__)
print("numpy version : " + np.__version__)

wm811k = pd.read_pickle("../data/LSWMD.pkl")

print(wm811k.head())

wm811k.info()

# 불필요 컬럼 제거
wm811k = wm811k.drop(['waferIndex'], axis=1)

print(wm811k.head())


# wafermap size 확인 및 컬럼 추가
def find_dim(x):
    dim0 = np.size(x, axis=0)
    dim1 = np.size(x, axis=1)
    return dim0, dim1


wm811k['waferMapDim'] = wm811k['waferMap'].apply(find_dim)
wm811k.head()

# 불량 클래스 확인 및 학습/검증/테스트 데이터 셋 확인
wm811k['failureNum'] = wm811k['failureType']
wm811k['trainTestNum'] = wm811k['trianTestLabel']
mapping_type = {'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 'Loc': 4, 'Random': 5, 'Scratch': 6, 'Near-full': 7, 'none': 8}
mapping_traintest = {'Training': 0, 'Test': 1}
wm811k = wm811k.replace({'failureNum': mapping_type, 'trainTestNum': mapping_traintest})
wm811k.head()

wm811k['trianTestLabel'].apply(lambda x: str(x)).value_counts()

wm811k['trainTestNum'].apply(lambda x: str(x)).value_counts()

wm811k_train = wm811k.query("trainTestNum == 0")

wm811k_train.info()

# 학습 데이터 내 불량 클래스 개수 확인
wm811k_train['failureNum'].value_counts()

'''
- Augmentation 구현
  : 기법 별로 함수 작성
  : 비율 개수만큼 sampling
  : sampling 데이터에 작성한 기법별 함수 적용
  : waferMapDim 확인 / waferMap 이미지 부분 잘리거나 한 부분 없게끔 적용되게 확인 : 미리 padding 주기?
  : data concat
'''
'''
def test_rotation_10_degree(data_img):
    height, width = data_img.shape
    # positive for anti-clockwise and negative for clockwise
    rotation_10_degree_img = cv2.getRotationMatrix2D((width/2, height/2), 10, 1) # 중심점, 각도, 배율
    dst = cv2.warpAffine(data_img, rotation_10_degree_img, (width,height))
    return dst

def test_rotation_minus_10_degree(data_img):
    height, width = data_img.shape
    # positive for anti-clockwise and negative for clockwise
    rotation_10_degree_img = cv2.getRotationMatrix2D((width/2, height/2), -10, 1) # 중심점, 각도, 배율
    dst = cv2.warpAffine(data_img, rotation_10_degree_img, (width,height))
    return dst

#평행이동
def test_translate(data_img):
    height, width = data_img.shape
    # random ~ 범위 지정하여 함수 실행 시 마다 무작위로 평행이동 정도 부여하게끔 수정 필요
    # 224x224 size 안에서 이동하도록 코딩 필요
    translate_matrix = np.float32([[1,0,10], [0,1,5]]) # 세로 10, 가로 5 만큼 평행 이동
    dst = cv2.warpAffine(data_img, translate_matrix, (width,height))
    return dst

def test_shearing(data_img):
    height, width = data_img.shape
    # random ~ 범위 지정하여 함수 실행 시 마다 x축, y축 shearing 정도 부여하게끔 수정 필요
    shearing_matrix = np.float32([[1, 0.5, 0],    # shearing applied to y-axis
             	                  [0, 1  , 0],    # M = np.float32([[1,   0, 0],
            	                  [0, 0  , 1]])   #             	[0.5, 1, 0],
                                                  #             	[0,   0, 1]])
    # apply a perspective transformation to the image
    dst = cv2.warpPerspective(data_img, shearing_matrix, (int(width*1.5),int(height*1.5)))
    # 변형 이미지 중심 보정?
    return dst

def test_zoom(data_img):
    height, width = data_img.shape
    # 2배 확대 이미지 / 가로 세로 값이 조건에 맞으면 dstsize로 값 부여하여 세밀 조정 가능
    # 224x224 size 안에서 2배 확대 가능하도록 코딩 필요
    dst = cv2.pyrUp(data_img, dstsize=(width * 2, height * 2), borderType=cv2.BORDER_DEFAULT)
    return dst

def test_filping(data_img):
    height, width = data_img.shape
    dst = cv2.flip(data_img, random.choice([0, 1])) # 0:상하 반전, 1:좌우 반전
    return dst

test_wafermap = wm811k_train.query("failureNum == 6").sample(n=5)

test_wafermap.head()

test_img = test_wafermap.iloc[3]["waferMap"]
plt.imshow(test_img)
plt.show()

plt.imshow(test_rotation_10_degree(test_img))
plt.show()

plt.imshow(test_rotation_minus_10_degree(test_img))
plt.show()

plt.imshow(test_translate(test_img))
plt.show()

plt.imshow(test_shearing(test_img))
plt.show()

plt.imshow(test_zoom(test_img))
plt.show()

plt.imshow(test_filping(test_img))
plt.show()
'''
#Input size에 맞게 zero-padding
def zero_padding(data_img, set_size):
    '''
    height, width = data_img.shape

    if max(height, width) > set_size:
        return data_img

    delta_width = set_size - width
    delta_height = set_size - height
    top, bottom = delta_height // 2, delta_height - (delta_height // 2)
    left, right = delta_width // 2, delta_width - (delta_width // 2)
    '''
    #padded_img = cv2.copyMakeBorder(data_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    padded_img = cv2.resize(data_img, (set_size, set_size), interpolation=cv2.INTER_LINEAR)
    return padded_img
'''
# 원본
test_origin = test_wafermap.iloc[2]["waferMap"]
plt.imshow(test_origin)
plt.show()

pad_origin = zero_padding(test_origin, 224)
plt.imshow(pad_origin)
plt.show()

# 10도 회전 :
degree_10 = test_rotation_10_degree(test_origin)
plt.imshow(degree_10)
plt.show()

pad_10_degree = zero_padding(degree_10, 224)
plt.imshow(pad_10_degree)
plt.show()

degree_10_pad_origin = test_rotation_10_degree(pad_origin)
plt.imshow(degree_10_pad_origin)
plt.show()

#평행이동
translated =  test_translate(test_origin)
plt.imshow(translated)
plt.show()

pad_translated = zero_padding(translated, 224)
plt.imshow(pad_translated)
plt.show()

translate_pad_origin = test_translate(pad_origin)
plt.imshow(translate_pad_origin)
plt.show()

translate_pad_origin.shape

shearing : 잘리지 않을 범위 내에서 shearing 이후 pad 하는 것이 나을 듯
shearing 정도에 따라 translate와 마찬가지로 wafer 모양이 잘릴 가능성이 있음
pad 이후 shearing 시 shape이 변함

sheared = test_shearing(test_origin)
plt.imshow(sheared)
plt.show()

pad_sheared = zero_padding(sheared, 224)
plt.imshow(pad_sheared)
plt.show()

sheared_pad_origin = test_shearing(pad_origin)
plt.imshow(sheared_pad_origin)
plt.show()

sheared_pad_origin.shape

#zooming : pad 이후 zoom할 경우 448x448 사이즈 이미지가 생성되므로 zoom 이후 pad 적용
zoomed = test_zoom(test_origin)
plt.imshow(zoomed)
plt.show()

pad_zoomed = zero_padding(zoomed, 224)
plt.imshow(pad_zoomed)
plt.show()

zoom_pad_origin = test_zoom(pad_origin)
plt.imshow(zoom_pad_origin)
plt.show()
'''
'''
augmentaion 함수별 random 요소 추가 및 padding 함수 추가
rotation : 회전 변환 이후 padding
translate : padding 이후 평행이동, dsize ~ 224로 출력
shearing : shearing 정도에 random 요소 추가, dsize ~ 224로 출력
'''

def rotation_10_degree(data_img):
    height, width = data_img.shape
    rotation_10_degree_img = cv2.getRotationMatrix2D((width/2, height/2), 10, 1)
    dst = cv2.warpAffine(data_img, rotation_10_degree_img, (width,height))
    padded_dst = zero_padding(dst, 224)
    return padded_dst

def rotation_minus_10_degree(data_img):
    height, width = data_img.shape
    rotation_10_degree_img = cv2.getRotationMatrix2D((width/2, height/2), -10, 1)
    dst = cv2.warpAffine(data_img, rotation_10_degree_img, (width,height))
    padded_dst = zero_padding(dst, 224)
    return padded_dst

def translate(data_img):
    padded_img = zero_padding(data_img, 224)
    x_translate = random.randrange(-20, 21)
    y_translate = random.randrange(-20, 21)
    translate_matrix = np.float32([[1, 0, y_translate],
                                   [0, 1, x_translate]])
    dst = cv2.warpAffine(padded_img, translate_matrix, (224,224))
    return dst

def filping(data_img):
    padded_img = zero_padding(data_img, 224)
    dst = cv2.flip(padded_img, random.choice([0, 1]))
    return dst

def shearing(data_img):
    x_shearing = random.random()
    y_shearing = random.random()
    shearing_matrix = np.float32([[1,          x_shearing, 0],
             	                  [y_shearing, 1,          0],
            	                  [0,          0  ,        1]])
    dst = cv2.warpPerspective(data_img, shearing_matrix, (224, 224))
    return dst

def resizing(data_img):
    scale_list = [0.5, 0.6, 0.7, 0.8, 1.05]
    dst = cv2.resize(data_img, dsize=(0,0), fx=random.choice(scale_list), fy=random.choice(scale_list),
                     interpolation=cv2.INTER_LINEAR)
    padded_img = zero_padding(dst, 224)
    return padded_img

'''
Data-Augmentaion - 클래스별로 10,000개 / 논문 Augmentation 기법 적용
10도 회전 : 20%
좌우 대칭 및 너비 이동(horizontal flipping and width shift) : 20%
높이 이동(height shfit) : 15%
전단 범위(shearing range) : 10%
채널이동 및 확대 축소(channel shift and zooming) : 10%
75%밖에 안 되는 듯 : 증량 비율 수정
'''

# None(8): 36,730 중 10,000 개 Sampling
# 참고 : https://rfriend.tistory.com/602
wm811k_new_train_class_None = wm811k_train.query("failureNum == 8").sample(n=10000, random_state=2022)
wm811k_new_train_class_None["waferMap_augmentation"] = wm811k_new_train_class_None["waferMap"].apply(lambda x: zero_padding(x,224))
wm811k_new_train_class_None['waferMap_augmentation_Dim']=wm811k_new_train_class_None['waferMap_augmentation'].apply(find_dim)
wm811k_new_train_class_None.info()

wm811k_new_train_class_None['waferMap_augmentation_Dim'].apply(lambda x: str(x)).value_counts()

# Edge-Ring(3) 현 보유 8,554개 / 1,446개 augmentation 필요
wm811k_new_train_class_Edge_Ring = wm811k_train.query("failureNum == 3")
wm811k_new_train_class_Edge_Ring["waferMap_augmentation"] = wm811k_new_train_class_Edge_Ring['waferMap'].apply(lambda x: zero_padding(x,224))
wm811k_new_train_class_Edge_Ring['waferMap_augmentation_Dim']=wm811k_new_train_class_Edge_Ring['waferMap_augmentation'].apply(find_dim)
wm811k_new_train_class_Edge_Ring.info()

# +10도 회전 : 10% - 144개
wm811k_new_train_class_Edge_Ring_10 = wm811k_new_train_class_Edge_Ring.sample(n=144, random_state=2022)
wm811k_new_train_class_Edge_Ring_10["waferMap_augmentation"] = wm811k_new_train_class_Edge_Ring_10["waferMap"].apply(lambda x: rotation_10_degree(x))
wm811k_new_train_class_Edge_Ring_10['waferMap_augmentation_Dim']=wm811k_new_train_class_Edge_Ring_10['waferMap_augmentation'].apply(find_dim)

# -10도 회전 : 10% - 144개
wm811k_new_train_class_Edge_Ring_minus_10 = wm811k_new_train_class_Edge_Ring.sample(n=144, random_state=2022)
wm811k_new_train_class_Edge_Ring_minus_10["waferMap_augmentation"] = wm811k_new_train_class_Edge_Ring_minus_10["waferMap"].apply(lambda x: rotation_minus_10_degree(x))
wm811k_new_train_class_Edge_Ring_minus_10['waferMap_augmentation_Dim']=wm811k_new_train_class_Edge_Ring_minus_10['waferMap_augmentation'].apply(find_dim)

# 좌우 대칭 : 20%  - 288개
wm811k_new_train_class_Edge_Ring_flip = wm811k_new_train_class_Edge_Ring.sample(n=288, random_state=2022)
wm811k_new_train_class_Edge_Ring_flip["waferMap_augmentation"] = wm811k_new_train_class_Edge_Ring_flip["waferMap"].apply(lambda x: filping(x))
wm811k_new_train_class_Edge_Ring_flip['waferMap_augmentation_Dim']=wm811k_new_train_class_Edge_Ring_flip['waferMap_augmentation'].apply(find_dim)

# 평행 이동 : 30% - 432개
wm811k_new_train_class_Edge_Ring_translate = wm811k_new_train_class_Edge_Ring.sample(n=432, random_state=2022)
wm811k_new_train_class_Edge_Ring_translate["waferMap_augmentation"] = wm811k_new_train_class_Edge_Ring_translate["waferMap"].apply(lambda x: translate(x))
wm811k_new_train_class_Edge_Ring_translate['waferMap_augmentation_Dim']=wm811k_new_train_class_Edge_Ring_translate['waferMap_augmentation'].apply(find_dim)

# 전단 범위(shearing range) : 10% - 144개
wm811k_new_train_class_Edge_Ring_shearing = wm811k_new_train_class_Edge_Ring.sample(n=144, random_state=2022)
wm811k_new_train_class_Edge_Ring_shearing["waferMap_augmentation"] = wm811k_new_train_class_Edge_Ring_shearing["waferMap"].apply(lambda x: shearing(x))
wm811k_new_train_class_Edge_Ring_shearing['waferMap_augmentation_Dim']=wm811k_new_train_class_Edge_Ring_shearing['waferMap_augmentation'].apply(find_dim)

# 확대 : 20% - 294개
wm811k_new_train_class_Edge_Ring_resize = wm811k_new_train_class_Edge_Ring.sample(n=294, random_state=2022)
wm811k_new_train_class_Edge_Ring_resize["waferMap_augmentation"] = wm811k_new_train_class_Edge_Ring_resize["waferMap"].apply(lambda x: resizing(x))
wm811k_new_train_class_Edge_Ring_resize['waferMap_augmentation_Dim']=wm811k_new_train_class_Edge_Ring_resize['waferMap_augmentation'].apply(find_dim)

# concat
wm811k_new_train_class_Edge_Ring_augmentation = pd.concat([wm811k_new_train_class_Edge_Ring,
                                                           wm811k_new_train_class_Edge_Ring_10,
                                                           wm811k_new_train_class_Edge_Ring_minus_10,
                                                           wm811k_new_train_class_Edge_Ring_flip,
                                                           wm811k_new_train_class_Edge_Ring_translate,
                                                           wm811k_new_train_class_Edge_Ring_shearing,
                                                           wm811k_new_train_class_Edge_Ring_resize
                                                          ])
wm811k_new_train_class_Edge_Ring_augmentation.info()

# Center(0) 현 보유 3,462개 / 6,538개 augmentation 필요
wm811k_new_train_class_Center = wm811k_train.query("failureNum == 0")
wm811k_new_train_class_Center[ "waferMap_augmentation"] = wm811k_new_train_class_Center['waferMap'].apply(lambda x: zero_padding(x,224))
wm811k_new_train_class_Center[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Center['waferMap_augmentation'].apply(find_dim)
wm811k_new_train_class_Center.info()

# +10도 회전 : 10% - 654개
wm811k_new_train_class_Center_10 = wm811k_new_train_class_Center.sample(n=654, random_state=2022)
wm811k_new_train_class_Center_10[ "waferMap_augmentation"] = wm811k_new_train_class_Center_10["waferMap"].apply(lambda x: rotation_10_degree(x))
wm811k_new_train_class_Center_10[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Center_10['waferMap_augmentation'].apply(find_dim)

# -10도 회전 : 10% - 654개
wm811k_new_train_class_Center_minus_10 = wm811k_new_train_class_Center.sample(n=654, random_state=2022)
wm811k_new_train_class_Center_minus_10[ "waferMap_augmentation"] = wm811k_new_train_class_Center_minus_10["waferMap"].apply(lambda x: rotation_minus_10_degree(x))
wm811k_new_train_class_Center_minus_10[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Center_minus_10['waferMap_augmentation'].apply(find_dim)

# 좌우 대칭 : 20%  - 1308개
wm811k_new_train_class_Center_flip = wm811k_new_train_class_Center.sample(n=1308, random_state=2022)
wm811k_new_train_class_Center_flip[ "waferMap_augmentation"] = wm811k_new_train_class_Center_flip["waferMap"].apply(lambda x: filping(x))
wm811k_new_train_class_Center_flip[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Center_flip['waferMap_augmentation'].apply(find_dim)

# 평행 이동 : 30%  - 1962개
wm811k_new_train_class_Center_translate = wm811k_new_train_class_Center.sample(n=1962, random_state=2022)
wm811k_new_train_class_Center_translate[ "waferMap_augmentation"] = wm811k_new_train_class_Center_translate["waferMap"].apply(lambda x: translate(x))
wm811k_new_train_class_Center_translate[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Center_translate['waferMap_augmentation'].apply(find_dim)

# 전단 범위(shearing range) : 10% - 652개
wm811k_new_train_class_Center_shearing = wm811k_new_train_class_Center.sample(n=652, random_state=2022)
wm811k_new_train_class_Center_shearing[ "waferMap_augmentation"] = wm811k_new_train_class_Center_shearing["waferMap"].apply(lambda x: shearing(x))
wm811k_new_train_class_Center_shearing[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Center_shearing['waferMap_augmentation'].apply(find_dim)

# 확대 : 20% - 1308개
wm811k_new_train_class_Center_resize = wm811k_new_train_class_Center.sample(n=1308, random_state=2022)
wm811k_new_train_class_Center_resize[ "waferMap_augmentation"] = wm811k_new_train_class_Center_resize["waferMap"].apply(lambda x: resizing(x))
wm811k_new_train_class_Center_resize[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Center_resize['waferMap_augmentation'].apply(find_dim)

# concat
wm811k_new_train_class_Center_augmentation = pd.concat([wm811k_new_train_class_Center,
                                                        wm811k_new_train_class_Center_10,
                                                        wm811k_new_train_class_Center_minus_10,
                                                        wm811k_new_train_class_Center_flip,
                                                        wm811k_new_train_class_Center_translate,
                                                        wm811k_new_train_class_Center_shearing,
                                                        wm811k_new_train_class_Center_resize
                                                       ])
wm811k_new_train_class_Center_augmentation.info()

# Edge-Loc(2) 현 보유 2,417개 / 7,583개 augmentation 필요
wm811k_new_train_class_Edge_Loc = wm811k_train.query("failureNum == 2")
wm811k_new_train_class_Edge_Loc[ "waferMap_augmentation"] = wm811k_new_train_class_Edge_Loc['waferMap'].apply(lambda x: zero_padding(x,224))
wm811k_new_train_class_Edge_Loc[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Edge_Loc['waferMap_augmentation'].apply(find_dim)
wm811k_new_train_class_Edge_Loc.info()

# +10도 회전 : 10% - 758개
wm811k_new_train_class_Edge_Loc_10 = wm811k_new_train_class_Edge_Loc.sample(n=758, random_state=2022)
wm811k_new_train_class_Edge_Loc_10[ "waferMap_augmentation"] = wm811k_new_train_class_Edge_Loc_10["waferMap"].apply(lambda x: rotation_10_degree(x))
wm811k_new_train_class_Edge_Loc_10[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Edge_Loc_10['waferMap_augmentation'].apply(find_dim)

# -10도 회전 : 10% - 758개
wm811k_new_train_class_Edge_Loc_minus_10 = wm811k_new_train_class_Edge_Loc.sample(n=758, random_state=2022)
wm811k_new_train_class_Edge_Loc_minus_10[ "waferMap_augmentation"] = wm811k_new_train_class_Edge_Loc_minus_10["waferMap"].apply(lambda x: rotation_minus_10_degree(x))
wm811k_new_train_class_Edge_Loc_minus_10[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Edge_Loc_minus_10['waferMap_augmentation'].apply(find_dim)

# 좌우 대칭 : 20% - 1517개
wm811k_new_train_class_Edge_Loc_flip = wm811k_new_train_class_Edge_Loc.sample(n=1517, random_state=2022)
wm811k_new_train_class_Edge_Loc_flip[ "waferMap_augmentation"] = wm811k_new_train_class_Edge_Loc_flip["waferMap"].apply(lambda x: filping(x))
wm811k_new_train_class_Edge_Loc_flip[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Edge_Loc_flip['waferMap_augmentation'].apply(find_dim)

# 평행 이동 : 30% - 2275개
wm811k_new_train_class_Edge_Loc_translate = wm811k_new_train_class_Edge_Loc.sample(n=2275, random_state=2022)
wm811k_new_train_class_Edge_Loc_translate[ "waferMap_augmentation"] = wm811k_new_train_class_Edge_Loc_translate["waferMap"].apply(lambda x: translate(x))
wm811k_new_train_class_Edge_Loc_translate[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Edge_Loc_translate['waferMap_augmentation'].apply(find_dim)

# 전단 범위(shearing range) : 10% - 758개
wm811k_new_train_class_Edge_Loc_shearing = wm811k_new_train_class_Edge_Loc.sample(n=758, random_state=2022)
wm811k_new_train_class_Edge_Loc_shearing[ "waferMap_augmentation"] = wm811k_new_train_class_Edge_Loc_shearing["waferMap"].apply(lambda x: shearing(x))
wm811k_new_train_class_Edge_Loc_shearing[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Edge_Loc_shearing['waferMap_augmentation'].apply(find_dim)

# 확대 : 20% - 1517개
wm811k_new_train_class_Edge_Loc_resize = wm811k_new_train_class_Edge_Loc.sample(n=1517, random_state=2022)
wm811k_new_train_class_Edge_Loc_resize[ "waferMap_augmentation"] = wm811k_new_train_class_Edge_Loc_resize["waferMap"].apply(lambda x: resizing(x))
wm811k_new_train_class_Edge_Loc_resize[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Edge_Loc_resize['waferMap_augmentation'].apply(find_dim)

# concat
wm811k_new_train_class_Edge_Loc_augmentation = pd.concat([wm811k_new_train_class_Edge_Loc,
                                                          wm811k_new_train_class_Edge_Loc_10,
                                                          wm811k_new_train_class_Edge_Loc_minus_10,
                                                          wm811k_new_train_class_Edge_Loc_flip,
                                                          wm811k_new_train_class_Edge_Loc_translate,
                                                          wm811k_new_train_class_Edge_Loc_shearing,
                                                          wm811k_new_train_class_Edge_Loc_resize
                                                         ])
wm811k_new_train_class_Edge_Loc_augmentation.info()

# Loc(4) 현 보유 1,620개 / 8,380개 augmentation 필요
wm811k_new_train_class_Loc = wm811k_train.query("failureNum == 4")
wm811k_new_train_class_Loc[ "waferMap_augmentation"] = wm811k_new_train_class_Loc['waferMap'].apply(lambda x: zero_padding(x,224))
wm811k_new_train_class_Loc[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Loc['waferMap_augmentation'].apply(find_dim)
wm811k_new_train_class_Loc.info()

# +10도 회전 : 10% - 838개
wm811k_new_train_class_Loc_10 = wm811k_new_train_class_Loc.sample(n=838, random_state=2022)
wm811k_new_train_class_Loc_10[ "waferMap_augmentation"] = wm811k_new_train_class_Loc_10["waferMap"].apply(lambda x: rotation_10_degree(x))
wm811k_new_train_class_Loc_10[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Loc_10['waferMap_augmentation'].apply(find_dim)

# -10도 회전 : 10% - 838개
wm811k_new_train_class_Loc_minus_10 = wm811k_new_train_class_Loc.sample(n=838, random_state=2022)
wm811k_new_train_class_Loc_minus_10[ "waferMap_augmentation"] = wm811k_new_train_class_Loc_minus_10["waferMap"].apply(lambda x: rotation_minus_10_degree(x))
wm811k_new_train_class_Loc_minus_10[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Loc_minus_10['waferMap_augmentation'].apply(find_dim)

# 중간 concat
wm811k_new_train_class_Loc_patial_concat = pd.concat([wm811k_new_train_class_Loc,
                                                      wm811k_new_train_class_Loc_10,
                                                      wm811k_new_train_class_Loc_minus_10
                                                     ])
wm811k_new_train_class_Loc_patial_concat.info()

# 좌우 대칭 : 20%  - "1676개"
wm811k_new_train_class_Loc_flip = wm811k_new_train_class_Loc_patial_concat.sample(n=1676, random_state=2022)
wm811k_new_train_class_Loc_flip[ "waferMap_augmentation"] = wm811k_new_train_class_Loc_flip["waferMap"].apply(lambda x: filping(x))
wm811k_new_train_class_Loc_flip[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Loc_flip['waferMap_augmentation'].apply(find_dim)

# 평행 이동 : 30%  - "2514개"
wm811k_new_train_class_Loc_translate = wm811k_new_train_class_Loc_patial_concat.sample(n=2514, random_state=2022)
wm811k_new_train_class_Loc_translate[ "waferMap_augmentation"] = wm811k_new_train_class_Loc_translate["waferMap"].apply(lambda x: translate(x))
wm811k_new_train_class_Loc_translate[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Loc_translate['waferMap_augmentation'].apply(find_dim)

# 전단 범위(shearing range) : 10% - 838개
wm811k_new_train_class_Loc_shearing = wm811k_new_train_class_Loc.sample(n=838, random_state=2022)
wm811k_new_train_class_Loc_shearing[ "waferMap_augmentation"] = wm811k_new_train_class_Loc_shearing["waferMap"].apply(lambda x: shearing(x))
wm811k_new_train_class_Loc_shearing[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Loc_shearing['waferMap_augmentation'].apply(find_dim)

# 확대 : 20% - "1676개"
wm811k_new_train_class_Loc_resize = wm811k_new_train_class_Loc_patial_concat.sample(n=1676, random_state=2022)
wm811k_new_train_class_Loc_resize[ "waferMap_augmentation"] = wm811k_new_train_class_Loc_resize["waferMap"].apply(lambda x: resizing(x))
wm811k_new_train_class_Loc_resize[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Loc_resize['waferMap_augmentation'].apply(find_dim)

# concat
wm811k_new_train_class_Loc_augmentation = pd.concat([wm811k_new_train_class_Loc,
                                                      wm811k_new_train_class_Loc_10,
                                                      wm811k_new_train_class_Loc_minus_10,
                                                      wm811k_new_train_class_Loc_flip,
                                                      wm811k_new_train_class_Loc_translate,
                                                      wm811k_new_train_class_Loc_shearing,
                                                      wm811k_new_train_class_Loc_resize
                                                     ])
wm811k_new_train_class_Loc_augmentation.info()

# Random(5) 현 보유 609개 / 9,391개 augmentation 필요
wm811k_new_train_class_Random = wm811k_train.query("failureNum == 5")
wm811k_new_train_class_Random[ "waferMap_augmentation"] = wm811k_new_train_class_Random['waferMap'].apply(lambda x: zero_padding(x,224))
wm811k_new_train_class_Random[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Random['waferMap_augmentation'].apply(find_dim)
wm811k_new_train_class_Random.info()

# +10도 회전 : 609개
wm811k_new_train_class_Random_10 = wm811k_new_train_class_Random.copy()
wm811k_new_train_class_Random_10[ "waferMap_augmentation"] = wm811k_new_train_class_Random_10["waferMap"].apply(lambda x: rotation_10_degree(x))
wm811k_new_train_class_Random_10[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Random_10['waferMap_augmentation'].apply(find_dim)

# -10도 회전 : 609개
wm811k_new_train_class_Random_minus_10 = wm811k_new_train_class_Random.copy()
wm811k_new_train_class_Random_minus_10[ "waferMap_augmentation"] = wm811k_new_train_class_Random_minus_10["waferMap"].apply(lambda x: rotation_minus_10_degree(x))
wm811k_new_train_class_Random_minus_10[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Random_minus_10['waferMap_augmentation'].apply(find_dim)

# 중간 concat : 1827개
wm811k_new_train_class_Random_patial_concat1 = pd.concat([wm811k_new_train_class_Random,
                                                          wm811k_new_train_class_Random_10,
                                                          wm811k_new_train_class_Random_minus_10
                                                         ])
wm811k_new_train_class_Random_patial_concat1.info()

# 좌우 대칭 : 1827개
wm811k_new_train_class_Random_flip = wm811k_new_train_class_Random_patial_concat1.copy()
wm811k_new_train_class_Random_flip[ "waferMap_augmentation"] = wm811k_new_train_class_Random_flip["waferMap"].apply(lambda x: filping(x))
wm811k_new_train_class_Random_flip[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Random_flip['waferMap_augmentation'].apply(find_dim)

# 중간 concat : 3654개
wm811k_new_train_class_Random_patial_concat2 = pd.concat([wm811k_new_train_class_Random,
                                                          wm811k_new_train_class_Random_10,
                                                          wm811k_new_train_class_Random_minus_10,
                                                          wm811k_new_train_class_Random_flip
                                                         ])
wm811k_new_train_class_Random_patial_concat2.info()

# 평행 이동 : 2115개
wm811k_new_train_class_Random_translate = wm811k_new_train_class_Random_patial_concat2.sample(n=2115, random_state=2022)
wm811k_new_train_class_Random_translate[ "waferMap_augmentation"] = wm811k_new_train_class_Random_translate["waferMap"].apply(lambda x: translate(x))
wm811k_new_train_class_Random_translate[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Random_translate['waferMap_augmentation'].apply(find_dim)

# 전단 범위(shearing range) : 2115개
wm811k_new_train_class_Random_shearing = wm811k_new_train_class_Random_patial_concat2.sample(n=2115, random_state=2022)
wm811k_new_train_class_Random_shearing[ "waferMap_augmentation"] = wm811k_new_train_class_Random_shearing["waferMap"].apply(lambda x: shearing(x))
wm811k_new_train_class_Random_shearing[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Random_shearing['waferMap_augmentation'].apply(find_dim)

# 확대 : 2116개
wm811k_new_train_class_Random_resize = wm811k_new_train_class_Random_patial_concat2.sample(n=2116, random_state=2022)
wm811k_new_train_class_Random_resize[ "waferMap_augmentation"] = wm811k_new_train_class_Random_resize["waferMap"].apply(lambda x: resizing(x))
wm811k_new_train_class_Random_resize[ 'waferMap_augmentation_Dim']=wm811k_new_train_class_Random_resize['waferMap_augmentation'].apply(find_dim)

# concat
wm811k_new_train_class_Random_augmentation = pd.concat([wm811k_new_train_class_Random,
                                                        wm811k_new_train_class_Random_10,
                                                        wm811k_new_train_class_Random_minus_10,
                                                        wm811k_new_train_class_Random_flip,
                                                        wm811k_new_train_class_Random_translate,
                                                        wm811k_new_train_class_Random_shearing,
                                                        wm811k_new_train_class_Random_resize
                                                       ])
wm811k_new_train_class_Random_augmentation.info()

# Scratch(6) 현 보유 500개 / 9,500개 augmentation 필요
wm811k_new_train_class_Scratch = wm811k_train.query("failureNum == 6")
wm811k_new_train_class_Scratch["waferMap_augmentation"] = wm811k_new_train_class_Scratch['waferMap'].apply(lambda x: zero_padding(x,224))
wm811k_new_train_class_Scratch['waferMap_augmentation_Dim']=wm811k_new_train_class_Scratch['waferMap_augmentation'].apply(find_dim)
wm811k_new_train_class_Scratch.info()

# +10도 회전 : 500개
wm811k_new_train_class_Scratch_10 = wm811k_new_train_class_Scratch.copy()
wm811k_new_train_class_Scratch_10["waferMap_augmentation"] = wm811k_new_train_class_Scratch_10["waferMap"].apply(lambda x: rotation_10_degree(x))
wm811k_new_train_class_Scratch_10['waferMap_augmentation_Dim']=wm811k_new_train_class_Scratch_10['waferMap_augmentation'].apply(find_dim)

# -10도 회전 : 500개
wm811k_new_train_class_Scratch_minus_10 = wm811k_new_train_class_Scratch.copy()
wm811k_new_train_class_Scratch_minus_10["waferMap_augmentation"] = wm811k_new_train_class_Scratch_minus_10["waferMap"].apply(lambda x: rotation_minus_10_degree(x))
wm811k_new_train_class_Scratch_minus_10['waferMap_augmentation_Dim']=wm811k_new_train_class_Scratch_minus_10['waferMap_augmentation'].apply(find_dim)

# 중간 concat1 : 1500개
wm811k_new_train_class_Scratch_patial_concat1 = pd.concat([wm811k_new_train_class_Scratch,
                                                           wm811k_new_train_class_Scratch_10,
                                                           wm811k_new_train_class_Scratch_minus_10
                                                          ])
wm811k_new_train_class_Scratch_patial_concat1.info()

# 좌우 대칭 : 1500개
wm811k_new_train_class_Scratch_flip = wm811k_new_train_class_Scratch_patial_concat1.copy()
wm811k_new_train_class_Scratch_flip["waferMap_augmentation"] = wm811k_new_train_class_Scratch_flip["waferMap"].apply(lambda x: filping(x))
wm811k_new_train_class_Scratch_flip['waferMap_augmentation_Dim']=wm811k_new_train_class_Scratch_flip['waferMap_augmentation'].apply(find_dim)

# 중간 concat2 : 3000개
wm811k_new_train_class_Scratch_patial_concat2 = pd.concat([wm811k_new_train_class_Scratch,
                                                           wm811k_new_train_class_Scratch_10,
                                                           wm811k_new_train_class_Scratch_minus_10,
                                                           wm811k_new_train_class_Scratch_flip
                                                          ])
wm811k_new_train_class_Scratch_patial_concat2.info()

# 평행 이동 : 3000개
wm811k_new_train_class_Scratch_translate = wm811k_new_train_class_Scratch_patial_concat2.copy()
wm811k_new_train_class_Scratch_translate["waferMap_augmentation"] = wm811k_new_train_class_Scratch_translate["waferMap"].apply(lambda x: translate(x))
wm811k_new_train_class_Scratch_translate['waferMap_augmentation_Dim']=wm811k_new_train_class_Scratch_translate['waferMap_augmentation'].apply(find_dim)

# 중간 concat3 : 6000개
wm811k_new_train_class_Scratch_patial_concat3 = pd.concat([wm811k_new_train_class_Scratch,
                                                           wm811k_new_train_class_Scratch_10,
                                                           wm811k_new_train_class_Scratch_minus_10,
                                                           wm811k_new_train_class_Scratch_flip,
                                                           wm811k_new_train_class_Scratch_translate
                                                          ])
wm811k_new_train_class_Scratch_patial_concat3.info()

# 전단 범위(shearing range) : 2000개
wm811k_new_train_class_Scratch_shearing = wm811k_new_train_class_Scratch_patial_concat3.sample(n=2000, random_state=2022)
wm811k_new_train_class_Scratch_shearing["waferMap_augmentation"] = wm811k_new_train_class_Scratch_shearing["waferMap"].apply(lambda x: shearing(x))
wm811k_new_train_class_Scratch_shearing['waferMap_augmentation_Dim']=wm811k_new_train_class_Scratch_shearing['waferMap_augmentation'].apply(find_dim)

# 확대 : 2000개
wm811k_new_train_class_Scratch_resize = wm811k_new_train_class_Scratch_patial_concat3.sample(n=2000, random_state=2022)
wm811k_new_train_class_Scratch_resize["waferMap_augmentation"] = wm811k_new_train_class_Scratch_resize["waferMap"].apply(lambda x: resizing(x))
wm811k_new_train_class_Scratch_resize['waferMap_augmentation_Dim']=wm811k_new_train_class_Scratch_resize['waferMap_augmentation'].apply(find_dim)

# concat
wm811k_new_train_class_Scratch_augmentation = pd.concat([wm811k_new_train_class_Scratch,
                                                         wm811k_new_train_class_Scratch_10,
                                                         wm811k_new_train_class_Scratch_minus_10,
                                                         wm811k_new_train_class_Scratch_flip,
                                                         wm811k_new_train_class_Scratch_translate,
                                                         wm811k_new_train_class_Scratch_shearing,
                                                         wm811k_new_train_class_Scratch_resize
                                                        ])
wm811k_new_train_class_Scratch_augmentation.info()

# Donut(1) 현 보유 409개 / 9,591개 augmentation 필요
wm811k_new_train_class_Donut = wm811k_train.query("failureNum == 1")
wm811k_new_train_class_Donut["waferMap_augmentation"] = wm811k_new_train_class_Donut['waferMap'].apply(lambda x: zero_padding(x,224))
wm811k_new_train_class_Donut['waferMap_augmentation_Dim']=wm811k_new_train_class_Donut['waferMap_augmentation'].apply(find_dim)
wm811k_new_train_class_Donut.info()

# +10도 회전 : 409개
wm811k_new_train_class_Donut_10 = wm811k_new_train_class_Donut.copy()
wm811k_new_train_class_Donut_10["waferMap_augmentation"] = wm811k_new_train_class_Donut_10["waferMap"].apply(lambda x: rotation_10_degree(x))
wm811k_new_train_class_Donut_10['waferMap_augmentation_Dim']=wm811k_new_train_class_Donut_10['waferMap_augmentation'].apply(find_dim)

# -10도 회전 : 409개
wm811k_new_train_class_Donut_minus_10 = wm811k_new_train_class_Donut.copy()
wm811k_new_train_class_Donut_minus_10["waferMap_augmentation"] = wm811k_new_train_class_Donut_minus_10["waferMap"].apply(lambda x: rotation_minus_10_degree(x))
wm811k_new_train_class_Donut_minus_10['waferMap_augmentation_Dim']=wm811k_new_train_class_Donut_minus_10['waferMap_augmentation'].apply(find_dim)

# 중간 concat1 : 1227개
wm811k_new_train_class_Donut_patial_concat1 = pd.concat([wm811k_new_train_class_Donut,
                                                         wm811k_new_train_class_Donut_10,
                                                         wm811k_new_train_class_Donut_minus_10
                                                        ])
wm811k_new_train_class_Donut_patial_concat1.info()

# 좌우 대칭 : 1227개
wm811k_new_train_class_Donut_flip = wm811k_new_train_class_Donut_patial_concat1.copy()
wm811k_new_train_class_Donut_flip["waferMap_augmentation"] = wm811k_new_train_class_Donut_flip["waferMap"].apply(lambda x: filping(x))
wm811k_new_train_class_Donut_flip['waferMap_augmentation_Dim']=wm811k_new_train_class_Donut_flip['waferMap_augmentation'].apply(find_dim)

# 중간 concat2 : 2454개
wm811k_new_train_class_Donut_patial_concat2 = pd.concat([wm811k_new_train_class_Donut,
                                                         wm811k_new_train_class_Donut_10,
                                                         wm811k_new_train_class_Donut_minus_10,
                                                         wm811k_new_train_class_Donut_flip
                                                        ])
wm811k_new_train_class_Donut_patial_concat2.info()

# 평행 이동 : 2454개
wm811k_new_train_class_Donut_translate = wm811k_new_train_class_Donut_patial_concat2.copy()
wm811k_new_train_class_Donut_translate["waferMap_augmentation"] = wm811k_new_train_class_Donut_translate["waferMap"].apply(lambda x: translate(x))
wm811k_new_train_class_Donut_translate['waferMap_augmentation_Dim']=wm811k_new_train_class_Donut_translate['waferMap_augmentation'].apply(find_dim)

# 중간 concat3 : 4908개
wm811k_new_train_class_Donut_patial_concat3 = pd.concat([wm811k_new_train_class_Donut,
                                                         wm811k_new_train_class_Donut_10,
                                                         wm811k_new_train_class_Donut_minus_10,
                                                         wm811k_new_train_class_Donut_flip,
                                                         wm811k_new_train_class_Donut_translate
                                                        ])
wm811k_new_train_class_Donut_patial_concat3.info()

# 전단 범위(shearing range) : 4908개
wm811k_new_train_class_Donut_shearing = wm811k_new_train_class_Donut_patial_concat3.copy()
wm811k_new_train_class_Donut_shearing["waferMap_augmentation"] = wm811k_new_train_class_Donut_shearing["waferMap"].apply(lambda x: shearing(x))
wm811k_new_train_class_Donut_shearing['waferMap_augmentation_Dim']=wm811k_new_train_class_Donut_shearing['waferMap_augmentation'].apply(find_dim)

# 중간 concat4 : 9816개
wm811k_new_train_class_Donut_patial_concat4 = pd.concat([wm811k_new_train_class_Donut,
                                                         wm811k_new_train_class_Donut_10,
                                                         wm811k_new_train_class_Donut_minus_10,
                                                         wm811k_new_train_class_Donut_flip,
                                                         wm811k_new_train_class_Donut_translate,
                                                         wm811k_new_train_class_Donut_shearing
                                                        ])
wm811k_new_train_class_Donut_patial_concat4.info()

# 확대 : 184개
wm811k_new_train_class_Donut_resize = wm811k_new_train_class_Donut_patial_concat4.sample(n=184, random_state=2022)
wm811k_new_train_class_Donut_resize["waferMap_augmentation"] = wm811k_new_train_class_Donut_resize["waferMap"].apply(lambda x: resizing(x))
wm811k_new_train_class_Donut_resize['waferMap_augmentation_Dim']=wm811k_new_train_class_Donut_resize['waferMap_augmentation'].apply(find_dim)

# concat
wm811k_new_train_class_Donut_augmentation = pd.concat([wm811k_new_train_class_Donut,
                                                       wm811k_new_train_class_Donut_10,
                                                       wm811k_new_train_class_Donut_minus_10,
                                                       wm811k_new_train_class_Donut_flip,
                                                       wm811k_new_train_class_Donut_translate,
                                                       wm811k_new_train_class_Donut_shearing,
                                                       wm811k_new_train_class_Donut_resize
                                                      ])
wm811k_new_train_class_Donut_augmentation.info()

# Near-full(7) 현 보유 54개 / 9,946개 augmentation 필요
wm811k_new_train_class_Near_Full = wm811k_train.query("failureNum == 7")
wm811k_new_train_class_Near_Full["waferMap_augmentation"] = wm811k_new_train_class_Near_Full['waferMap'].apply(lambda x: zero_padding(x,224))
wm811k_new_train_class_Near_Full['waferMap_augmentation_Dim']=wm811k_new_train_class_Near_Full['waferMap_augmentation'].apply(find_dim)
wm811k_new_train_class_Near_Full.info()

# +10도 회전 : 54개
wm811k_new_train_class_Near_Full_10 = wm811k_new_train_class_Near_Full.copy()
wm811k_new_train_class_Near_Full_10["waferMap_augmentation"] = wm811k_new_train_class_Near_Full_10["waferMap"].apply(lambda x: rotation_10_degree(x))
wm811k_new_train_class_Near_Full_10['waferMap_augmentation_Dim']=wm811k_new_train_class_Near_Full_10['waferMap_augmentation'].apply(find_dim)

# -10도 회전 : 54개
wm811k_new_train_class_Near_Full_minus_10 = wm811k_new_train_class_Near_Full.copy()
wm811k_new_train_class_Near_Full_minus_10["waferMap_augmentation"] = wm811k_new_train_class_Near_Full_minus_10["waferMap"].apply(lambda x: rotation_minus_10_degree(x))
wm811k_new_train_class_Near_Full_minus_10['waferMap_augmentation_Dim']= wm811k_new_train_class_Near_Full_minus_10['waferMap_augmentation'].apply(find_dim)

# 중간 concat1 : 162개
wm811k_new_train_class_Near_Full_patial_concat1 = pd.concat([wm811k_new_train_class_Near_Full,
                                                             wm811k_new_train_class_Near_Full_10,
                                                             wm811k_new_train_class_Near_Full_minus_10
                                                            ])
wm811k_new_train_class_Near_Full_patial_concat1.info()

# 좌우 대칭 : 162개
wm811k_new_train_class_Near_Full_flip = wm811k_new_train_class_Near_Full_patial_concat1.copy()
wm811k_new_train_class_Near_Full_flip["waferMap_augmentation"] = wm811k_new_train_class_Near_Full_flip["waferMap"].apply(lambda x: filping(x))
wm811k_new_train_class_Near_Full_flip['waferMap_augmentation_Dim']=wm811k_new_train_class_Near_Full_flip['waferMap_augmentation'].apply(find_dim)

# 중간 concat2 : 324개
wm811k_new_train_class_Near_Full_patial_concat2 = pd.concat([wm811k_new_train_class_Near_Full,
                                                             wm811k_new_train_class_Near_Full_10,
                                                             wm811k_new_train_class_Near_Full_minus_10,
                                                             wm811k_new_train_class_Near_Full_flip
                                                            ])
wm811k_new_train_class_Near_Full_patial_concat2.info()

# 평행 이동 : 324개
wm811k_new_train_class_Near_Full_translate = wm811k_new_train_class_Near_Full_patial_concat2.copy()
wm811k_new_train_class_Near_Full_translate["waferMap_augmentation"] = wm811k_new_train_class_Near_Full_translate["waferMap"].apply(lambda x: translate(x))
wm811k_new_train_class_Near_Full_translate['waferMap_augmentation_Dim']= wm811k_new_train_class_Near_Full_translate['waferMap_augmentation'].apply(find_dim)

# 중간 concat3 : 648개
wm811k_new_train_class_Near_Full_patial_concat3 = pd.concat([wm811k_new_train_class_Near_Full,
                                                             wm811k_new_train_class_Near_Full_10,
                                                             wm811k_new_train_class_Near_Full_minus_10,
                                                             wm811k_new_train_class_Near_Full_flip,
                                                             wm811k_new_train_class_Near_Full_translate
                                                            ])
wm811k_new_train_class_Near_Full_patial_concat3.info()

# 전단 범위(shearing range) : 648개
wm811k_new_train_class_Near_Full_shearing = wm811k_new_train_class_Near_Full_patial_concat3.copy()
wm811k_new_train_class_Near_Full_shearing["waferMap_augmentation"] = wm811k_new_train_class_Near_Full_shearing["waferMap"].apply(lambda x: shearing(x))
wm811k_new_train_class_Near_Full_shearing['waferMap_augmentation_Dim']=wm811k_new_train_class_Near_Full_shearing['waferMap_augmentation'].apply(find_dim)

# 중간 concat4 : 1296개
wm811k_new_train_class_Near_Full_patial_concat4 = pd.concat([wm811k_new_train_class_Near_Full,
                                                             wm811k_new_train_class_Near_Full_10,
                                                             wm811k_new_train_class_Near_Full_minus_10,
                                                             wm811k_new_train_class_Near_Full_flip,
                                                             wm811k_new_train_class_Near_Full_translate,
                                                             wm811k_new_train_class_Near_Full_shearing
                                                            ])
wm811k_new_train_class_Near_Full_patial_concat4.info()

# 확대 : 1296개
wm811k_new_train_class_Near_Full_resize = wm811k_new_train_class_Near_Full_patial_concat4.copy()
wm811k_new_train_class_Near_Full_resize["waferMap_augmentation"] = wm811k_new_train_class_Near_Full_resize["waferMap"].apply(lambda x: resizing(x))
wm811k_new_train_class_Near_Full_resize['waferMap_augmentation_Dim'] = wm811k_new_train_class_Near_Full_resize['waferMap_augmentation'].apply(find_dim)

# 중간 concat5 : 2592개
wm811k_new_train_class_Near_Full_patial_concat5 = pd.concat([wm811k_new_train_class_Near_Full,
                                                             wm811k_new_train_class_Near_Full_10,
                                                             wm811k_new_train_class_Near_Full_minus_10,
                                                             wm811k_new_train_class_Near_Full_flip,
                                                             wm811k_new_train_class_Near_Full_translate,
                                                             wm811k_new_train_class_Near_Full_shearing,
                                                             wm811k_new_train_class_Near_Full_resize
                                                            ])
wm811k_new_train_class_Near_Full_patial_concat5.info()


# 남은 데이터 7408개 : size 변동의 위험이 있는 shearing과 resize를 제외한 rotation, flip, translate로 나머지 부족한 데이터 증강
# 10도 회전 추가
wm811k_new_train_class_Near_Full_10_2 = wm811k_new_train_class_Near_Full_patial_concat5.sample(n=1852, random_state=2022)
wm811k_new_train_class_Near_Full_10_2["waferMap_augmentation"] = wm811k_new_train_class_Near_Full_10_2["waferMap"].apply(lambda x: rotation_10_degree(x))
wm811k_new_train_class_Near_Full_10_2['waferMap_augmentation_Dim']=wm811k_new_train_class_Near_Full_10_2['waferMap_augmentation'].apply(find_dim)

# -10도 회전 추가
wm811k_new_train_class_Near_Full_minus_10_2 = wm811k_new_train_class_Near_Full_patial_concat5.sample(n=1852, random_state=2022)
wm811k_new_train_class_Near_Full_minus_10_2["waferMap_augmentation"] = wm811k_new_train_class_Near_Full_minus_10_2["waferMap"].apply(lambda x: rotation_minus_10_degree(x))
wm811k_new_train_class_Near_Full_minus_10_2['waferMap_augmentation_Dim']=wm811k_new_train_class_Near_Full_minus_10_2['waferMap_augmentation'].apply(find_dim)

# flip 추가
wm811k_new_train_class_Near_Full_flip_2 = wm811k_new_train_class_Near_Full_patial_concat5.sample(n=1852, random_state=2022)
wm811k_new_train_class_Near_Full_flip_2["waferMap_augmentation"] = wm811k_new_train_class_Near_Full_flip_2["waferMap"].apply(lambda x: filping(x))
wm811k_new_train_class_Near_Full_flip_2['waferMap_augmentation_Dim']=wm811k_new_train_class_Near_Full_flip_2['waferMap_augmentation'].apply(find_dim)


# 평행이동 추가
wm811k_new_train_class_Near_Full_translate_2 = wm811k_new_train_class_Near_Full_patial_concat5.sample(n=1852, random_state=2022)
wm811k_new_train_class_Near_Full_translate_2["waferMap_augmentation"] = wm811k_new_train_class_Near_Full_translate_2["waferMap"].apply(lambda x: translate(x))
wm811k_new_train_class_Near_Full_translate_2['waferMap_augmentation_Dim']=wm811k_new_train_class_Near_Full_translate_2['waferMap_augmentation'].apply(find_dim)

# concat
wm811k_new_train_class_Near_Full_augmentation = pd.concat([wm811k_new_train_class_Near_Full,
                                                           wm811k_new_train_class_Near_Full_10,
                                                           wm811k_new_train_class_Near_Full_minus_10,
                                                           wm811k_new_train_class_Near_Full_flip,
                                                           wm811k_new_train_class_Near_Full_translate,
                                                           wm811k_new_train_class_Near_Full_shearing,
                                                           wm811k_new_train_class_Near_Full_resize,
                                                           wm811k_new_train_class_Near_Full_10_2,
                                                           wm811k_new_train_class_Near_Full_minus_10_2,
                                                           wm811k_new_train_class_Near_Full_flip_2,
                                                           wm811k_new_train_class_Near_Full_translate_2,
                                                          ])
wm811k_new_train_class_Near_Full_augmentation.info()

# 데이터 size 확인 및 New train data concat
wm811k_new_train = pd.concat([wm811k_new_train_class_None,
                              wm811k_new_train_class_Edge_Ring_augmentation,
                              wm811k_new_train_class_Center_augmentation,
                              wm811k_new_train_class_Edge_Loc_augmentation,
                              wm811k_new_train_class_Loc_augmentation,
                              wm811k_new_train_class_Random_augmentation,
                              wm811k_new_train_class_Scratch_augmentation,
                              wm811k_new_train_class_Donut_augmentation,
                              wm811k_new_train_class_Near_Full_augmentation
                             ])

wm811k_new_train.info()

# size check
wm811k_new_train['waferMap_augmentation_Dim'].apply(lambda x: str(x)).value_counts()

wm811k_new_train['failureNum'].apply(lambda x: str(x)).value_counts()

wm811k_new_train_reshape = np.ones((90000, 224, 224))

for i in range(len(wm811k_new_train['waferMap_augmentation'])):
    if i % 1000 == 0:
        print(i)
    wm811k_new_train_reshape[i] = wm811k_new_train['waferMap_augmentation'].iloc[i]

wm811k_new_train_reshape.shape

save_test_none = wm811k_new_train_reshape[500] # wm811k_new_train_class_None,
save_test_edge_ring = wm811k_new_train_reshape[10500] # wm811k_new_train_class_Edge_Ring_augmentation,
save_test_center = wm811k_new_train_reshape[20500] # wm811k_new_train_class_Center_augmentation,
save_test_edge_loc = wm811k_new_train_reshape[30500] # wm811k_new_train_class_Edge_Loc_augmentation,
save_test_loc = wm811k_new_train_reshape[40500] # wm811k_new_train_class_Loc_augmentation,
save_test_random = wm811k_new_train_reshape[50500] # wm811k_new_train_class_Random_augmentation,
save_test_scratch = wm811k_new_train_reshape[60500] # wm811k_new_train_class_Scratch_augmentation,
save_test_donut = wm811k_new_train_reshape[70500] # wm811k_new_train_class_Donut_augmentation,
save_test_near_full = wm811k_new_train_reshape[80500] # wm811k_new_train_class_Near_Full_augmentation

# plt.imshow(save_test_none)
# plt.show()
# plt.imshow(save_test_edge_ring)
# plt.show()
# plt.imshow(save_test_center)
# plt.show()
# plt.imshow(save_test_edge_loc)
# plt.show()
# plt.imshow(save_test_loc)
# plt.show()
# plt.imshow(save_test_random)
# plt.show()
# plt.imshow(save_test_scratch)
# plt.show()
# plt.imshow(save_test_donut)
# plt.show()
# plt.imshow(save_test_near_full)
# plt.show()

input_shape = (224, 224, 1)
class_num = 9
KERNEL_SIZE = 3

input_layer = layers.Input(shape=input_shape) # Input 224x224
x = layers.Conv2D(16, kernel_size=KERNEL_SIZE, padding='valid')(input_layer) # 16, 3x3
x = layers.Activation(activation='relu')(x) # ReLU
x = layers.BatchNormalization()(x) # BatchNormalization
x = layers.MaxPool2D(pool_size=(2,2))(x) # Max-pooling
x = layers.Conv2D(16, kernel_size=KERNEL_SIZE, padding='same')(x) # 16, 3x3
x = layers.Activation(activation='relu')(x) # ReLU
x = layers.BatchNormalization()(x) # BatchNormalization

x = layers.Conv2D(32, kernel_size=KERNEL_SIZE, padding='same')(x) # 32, 3x3
x = layers.Activation(activation='relu')(x) # ReLU
x = layers.BatchNormalization()(x) # BatchNormalization
x = layers.MaxPool2D(pool_size=(2,2))(x) # Max-pooling
x = layers.Conv2D(32, kernel_size=KERNEL_SIZE, padding='same')(x) # 32, 3x3
x = layers.Activation(activation='relu')(x) # ReLU
x = layers.BatchNormalization()(x) # BatchNormalization

x = layers.Conv2D(64, kernel_size=KERNEL_SIZE, padding='same')(x) # 64, 3x3
x = layers.Activation(activation='relu')(x) # ReLU
x = layers.BatchNormalization()(x) # BatchNormalization
x = layers.MaxPool2D(pool_size=(2,2))(x) # Max-pooling
x = layers.Conv2D(64, kernel_size=KERNEL_SIZE, padding='same')(x) # 64, 3x3
x = layers.Activation(activation='relu')(x) # ReLU
x = layers.BatchNormalization()(x) # BatchNormalization

x = layers.Conv2D(128, kernel_size=KERNEL_SIZE, padding='same')(x) # 128, 3x3
x = layers.Activation(activation='relu')(x) # ReLU
x = layers.BatchNormalization()(x) # BatchNormalization
x = layers.MaxPool2D(pool_size=(2,2))(x) # Max-pooling
x = layers.Conv2D(128, kernel_size=KERNEL_SIZE, padding='same')(x) # 128, 3x3
x = layers.Activation(activation='relu')(x) # ReLU
x = layers.BatchNormalization()(x) # BatchNormalization

x = layers.SpatialDropout2D(0.2)(x) # Spartial Dropout 0.2
x = layers.MaxPool2D(pool_size=(2,2))(x) # Max-pooling
x = layers.Flatten()(x) # 4608 확인
x = layers.Dense(512, activation='relu')(x)
output_layer = layers.Dense(class_num, activation='softmax')(x)

CNN_WDI = Model(input_layer, output_layer)

CNN_WDI.summary()

wm811k_new_train_reshape = wm811k_new_train_reshape.astype(np.int16)
wm811k_y_train_set = wm811k_new_train['failureNum']
wm811k_y_train_set_array = np.asarray(wm811k_y_train_set).astype(np.int16)

# MemoryError 해결
del wm811k_y_train_set
import gc
gc.collect()
# RAM에서 여유 용량이 생길 때까지 잠시 대기

# train / validation / test set 설정
# 60 / 20 / 20
X_train, X_test, y_train, y_test = train_test_split(wm811k_new_train_reshape, wm811k_y_train_set_array, test_size=0.2, random_state=2022)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=2022)  # 20/80 = 0.25


model_id = 'cnn_wdi'

LEARNING_RATE = 0.005
BATCH_SIZE = 10
EPOCH = 20

CNN_WDI.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                metrics=['acc'])

checkpointer = ModelCheckpoint(filepath="{0}.keras".format(model_id), verbose=1, save_best_only=True)

hist = CNN_WDI.fit(X_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, shuffle=True, validation_data=(X_val, y_val), callbacks=[checkpointer])

gc.collect()

score = CNN_WDI.evaluate(X_test, y_test, verbose=0)
print('Test Loss : ', score[0])
print('Test Accuracy : ', score[1])

# loss graph
y_vloss = hist.history['val_loss']
y_loss = hist.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# acc graph
y_vacc = hist.history['val_acc']
y_acc = hist.history['acc']

x_len = np.arange(len(y_acc))
plt.plot(x_len, y_vacc, marker='.', c='red', label="Validation-set Acc")
plt.plot(x_len, y_acc, marker='.', c='blue', label="Train-set Acc")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

y_predict = CNN_WDI.predict(X_test)

y_test

# test set 내 클래스 개수
np.unique(y_test, return_counts = True)

y_predict[0]

y_predict_label = np.array([np.argmax(x) for x in y_predict])

y_predict_label.shape

y_predict_label

# # confusion matrix
# confusion_matrix(y_test, y_predict_label)
#
# plt.imshow(confusion_matrix(y_test, y_predict_label), interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('Confusion matrix')
# plt.colorbar()
# ticks = np.arange(9) #ticks : 클래스의 수
# plt.xticks(ticks, ticks)
# plt.yticks(ticks, ticks)
# plt.ylabel('True labels')
# plt.xlabel('Predicted labels')
# plt.show()
# confusion matrix

cm = confusion_matrix(y_test, y_predict_label)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()


from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

accuracy_score(y_test, y_predict_label)
precision_score(y_test, y_predict_label, average='macro')
recall_score(y_test, y_predict_label, average='macro')
f1_score(y_test, y_predict_label, average='macro')
print(classification_report(y_test, y_predict_label))

# 저장된 best model
best_model = tf.keras.models.load_model('cnn_wdi.keras')
score = best_model.evaluate(X_test, y_test, verbose=0)
print('Test Loss : ', score[0])
print('Test Accuracy : ', score[1])

y_predict_best = best_model.predict(X_test)
y_predict_best_label = np.array([np.argmax(x) for x in y_predict_best])

# # confusion matrix
# confusion_matrix(y_test, y_predict_best_label)
#
# plt.imshow(confusion_matrix(y_test, y_predict_best_label), interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('Confusion matrix')
# plt.colorbar()
# ticks = np.arange(9) #ticks : 클래스의 수
# plt.xticks(ticks, ticks)
# plt.yticks(ticks, ticks)
# plt.ylabel('True labels')
# plt.xlabel('Predicted labels')
# plt.show()

# confusion matrix
cm = confusion_matrix(y_test, y_predict_best_label)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

accuracy_score(y_test, y_predict_best_label)

precision_score(y_test, y_predict_best_label, average='macro')

recall_score(y_test, y_predict_best_label, average='macro')

f1_score(y_test, y_predict_best_label, average='macro')

print(classification_report(y_test, y_predict_best_label))