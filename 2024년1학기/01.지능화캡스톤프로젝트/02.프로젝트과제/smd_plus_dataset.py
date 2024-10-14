import os
import shutil

# 경로를 지정
path = "c:\\smd_plus"
format = "zip"
file_name = os.path.join(path,"smd_plus.zip")
output_dir = os.path.join(path,"dataset")

# dataset 디렉토리 존재하면 초기화
if os.path.exists(output_dir):
    # dataset 디렉토리 삭제
    shutil.rmtree(output_dir)
    print("dataset 디렉토리 삭제 완료")

# 압축 해제
shutil.unpack_archive(file_name, output_dir, format)
print("smd_plus.zip 압축 해제 완료")

Onshore_name = os.path.join(output_dir,"VIS_Onshore.zip")
Onboard_name = os.path.join(output_dir,"VIS_Onboard.zip")
# 압축 해제
shutil.unpack_archive(Onshore_name, output_dir, format)
print("VIS_Onshore.zip 압축 해제 완료")

shutil.unpack_archive(Onboard_name, output_dir, format)
print("VIS_Onboard.zip 압축 해제 완료")

avi_dir = os.path.join(output_dir,"Videos")
mat_dir = os.path.join(output_dir,"ObjectGT")

images_dir = os.path.join(output_dir,"images")
labels_dir = os.path.join(output_dir,"labels")

avi_train_path = os.path.join(avi_dir,"train")
avi_val_path = os.path.join(avi_dir,"val")

mat_train_path = os.path.join(mat_dir,"train")
mat_val_path = os.path.join(mat_dir,"val")

img_train_path = os.path.join(images_dir,"train")
img_val_path = os.path.join(images_dir,"val")

lab_train_path = os.path.join(labels_dir,"train")
lab_val_path = os.path.join(labels_dir,"val")

#labels train/val 디렉토리 생성
os.makedirs(lab_train_path, exist_ok=True)
os.makedirs(lab_val_path, exist_ok=True)

#avi train/val 디렉토리 생성
os.makedirs(avi_train_path, exist_ok=True)
os.makedirs(avi_val_path, exist_ok=True)

shutil.move(avi_dir+"\\MVI_1469_VIS.avi", avi_val_path)
shutil.move(avi_dir+"\\MVI_1474_VIS.avi", avi_val_path)
shutil.move(avi_dir+"\\MVI_1587_VIS.avi", avi_val_path)
shutil.move(avi_dir+"\\MVI_1592_VIS.avi", avi_val_path)
shutil.move(avi_dir+"\\MVI_1613_VIS.avi", avi_val_path)
shutil.move(avi_dir+"\\MVI_1614_VIS.avi", avi_val_path)
shutil.move(avi_dir+"\\MVI_1615_VIS.avi", avi_val_path)
shutil.move(avi_dir+"\\MVI_1644_VIS.avi", avi_val_path)
shutil.move(avi_dir+"\\MVI_1645_VIS.avi", avi_val_path)
shutil.move(avi_dir+"\\MVI_1646_VIS.avi", avi_val_path)
shutil.move(avi_dir+"\\MVI_1448_VIS_Haze.avi", avi_val_path)
shutil.move(avi_dir+"\\MVI_1640_VIS.avi", avi_val_path)
shutil.move(avi_dir+"\\MVI_0799_VIS_OB.avi", avi_val_path)
shutil.move(avi_dir+"\\MVI_0804_VIS_OB.avi", avi_val_path)
print("avi파일 val 디렉토리 이동 완료")

#train avi파일 해당 디렉토리로 이동
files = os.listdir(avi_dir)
for file in files:
    if 'avi' in file:
        shutil.copy(avi_dir +"\\"+ file, avi_train_path +"\\"+ file)
        print('{} has been copied in new folder!'.format(file))

print("avi파일 train 디렉토리 복사 완료")

#mat train/val 디렉토리 생성
os.makedirs(mat_train_path, exist_ok=True)
os.makedirs(mat_val_path, exist_ok=True)

shutil.move(mat_dir+"\\MVI_1469_VIS_ObjectGT.mat", mat_val_path)
shutil.move(mat_dir+"\\MVI_1474_VIS_ObjectGT.mat", mat_val_path)
shutil.move(mat_dir+"\\MVI_1587_VIS_ObjectGT.mat", mat_val_path)
shutil.move(mat_dir+"\\MVI_1592_VIS_ObjectGT.mat", mat_val_path)
shutil.move(mat_dir+"\\MVI_1613_VIS_ObjectGT.mat", mat_val_path)
shutil.move(mat_dir+"\\MVI_1614_VIS_ObjectGT.mat", mat_val_path)
shutil.move(mat_dir+"\\MVI_1615_VIS_ObjectGT.mat", mat_val_path)
shutil.move(mat_dir+"\\MVI_1644_VIS_ObjectGT.mat", mat_val_path)
shutil.move(mat_dir+"\\MVI_1645_VIS_ObjectGT.mat", mat_val_path)
shutil.move(mat_dir+"\\MVI_1646_VIS_ObjectGT.mat", mat_val_path)
shutil.move(mat_dir+"\\MVI_1448_VIS_Haze_ObjectGT.mat", mat_val_path)
shutil.move(mat_dir+"\\MVI_1640_VIS_ObjectGT.mat", mat_val_path)
shutil.move(mat_dir+"\\MVI_0799_VIS_ObjectGT.mat", mat_val_path)
shutil.move(mat_dir+"\\MVI_0804_VIS_ObjectGT.mat", mat_val_path)
print("mat파일 val 디렉토리 이동 완료")

#train mat파일 해당 디렉토리로 이동
files = os.listdir(mat_dir)
for file in files:
    if 'mat' in file:
        shutil.copy(mat_dir +"\\"+ file, mat_train_path +"\\"+ file)
        print('{} has been copied in new folder!'.format(file))

print("mat파일 train 디렉토리 이동 완료")

"""**Convert video to .jpg**"""

import os
from os import listdir
from os.path import isfile, join
import cv2

def convert_videos_to_frames(video_files, output_folder):
    for video_file in video_files:
        video_name = os.path.basename(video_file).split('.')[0]
        video_name = video_name.replace('_OB', '').replace('_Haze', '')

        vidcap = cv2.VideoCapture(video_file)
        fps = vidcap.get(cv2.CAP_PROP_FPS)  # 원본 비디오의 프레임 속도를 가져옵니다.
        frame_rate = 30  # 원하는 프레임 속도 30FPS

        success, image = vidcap.read()
        count = 0
        frame_count = 0
        while success:
            if frame_count % int(fps / frame_rate) == 0:
                cv2.imwrite(join(output_folder, video_name + "_frame%d.jpg" % count), image)  # save frame as JPEG file
                count += 1
            success, image = vidcap.read()
            frame_count += 1

        print("Converted %d frames from video %s" % (count, video_name))

# Paths to video folders
train_path = avi_train_path
val_path = avi_val_path

# Get video files in each folder
train_videos = [join(train_path, f) for f in listdir(train_path) if isfile(join(train_path, f))]
val_videos = [join(val_path, f) for f in listdir(val_path) if isfile(join(val_path, f))]

# Output folders for frames
train_frames_path = img_train_path
val_frames_path = img_val_path

# Create output folders if they don't exist
os.makedirs(train_frames_path, exist_ok=True)
os.makedirs(val_frames_path, exist_ok=True)

# Convert train videos to frames
convert_videos_to_frames(train_videos, train_frames_path)

# Convert val videos to frames
convert_videos_to_frames(val_videos, val_frames_path)

# Count data in each folder
from os import listdir

train_count = len(listdir(train_frames_path))
val_count = len(listdir(val_frames_path))

print("Number of frames in Train folder:", train_count)
print("Number of frames in Val folder:", val_count)


"""**Convert .mat to .txt**"""

import cv2
import numpy as np
from scipy.io import loadmat
from os import listdir
from os.path import isfile, join


# 실제 .mat 파일에 대한 경로 설정
PATHS_TO_GT_FILES = [
    mat_train_path,
    mat_val_path
]

# 각 CSV 파일 세트에 대해 원하는 출력 경로 설정
OUTPUT_PATHS = [
    lab_train_path,
    lab_val_path
]

class Frame:
    """
    영상 프레임별로 데이터를 저장하는 클래스입니다.
    """
    def __init__(self, frame, image_name, bb, objects, motion, distance):
        """
        Parameters
        ----------
        frame : the frame number of the video. (string or int)
        image_name : the name of the image (for identification). (string)
        bb : bounding box coordinates of the objects. This is an array.
             Each line is the bb of an object and corresponds to
             [x_min, y_min, width, height]. See the dataset webpage for more info.
        objects : the type of objects. (array)
        motion : whether the objects are moving or not. (array)
        distance : distance of each object. (array)
        """
        self.frame = frame
        self.image_name = image_name
        self.bb = bb
        self.objects = objects
        self.motion = motion
        self.distance = distance

    def generate_yolo(self, output_path):
        """
        프레임에 대한 YOLO 파일을 생성합니다.

        Parameters
        ----------
        output_path : the desired path for the YOLO file. (string)
        """
        img_path = join(output_path, self.image_name)
        img = cv2.imread(img_path)
        img_height, img_width, _ = img.shape
        print("generate_yolo[img_path]:" + img_path)

        yolo_filename = join(output_path, self.image_name.replace(".jpg", ".txt"))
        yolo_filename = yolo_filename.replace("images","labels")
        print("generate_yolo[yolo_filename]:" + yolo_filename)
        with open(yolo_filename, "w") as f:
            if self.objects is not None and len(self.objects) > 0:
                for i in range(len(self.objects)):
                    if len(self.objects[i]) > 0 and int(self.objects[i][0]) != 0:
                        class_id = int(self.objects[i][0]) - 1
                        x_min = self.bb[i, 0]
                        y_min = self.bb[i, 1]
                        width = self.bb[i, 2]
                        height = self.bb[i, 3]

                        # YOLO 형식 좌표 계산
                        x_center = (x_min + width / 2) / img_width
                        y_center = (y_min + height / 2) / img_height
                        width /= img_width
                        height /= img_height

                        # YOLO 형식 라인 작성
                        yolo_line = (
                            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        )
                        f.write("%s\n" % yolo_line)

def generate_gt_files_dict(path_to_gt_files):
    """
    모든 정답 파일 위치가 포함된 사전을 만듭니다.

    Parameters
    ----------
    path_to_gt_files : the path to the ground truth files. (string)

    Returns
    -------
    object_gt_files_dict : dictionary in the form:
                           (key:value) -> (<video_name>:<video_path>)
    """
    object_gt_files_dict = {}
    for f in listdir(path_to_gt_files):
        if isfile(join(path_to_gt_files, f)):
            video_name = f.split(".")[0].replace("_Haze", "").replace("_ObjectGT", "").replace("_OB", "")
            object_gt_files_dict[video_name] = join(path_to_gt_files, f)

    return object_gt_files_dict

def load_mat_files_in_dict(path):
    """
    싱가포르 해양 데이터 세트의 모든 .mat 파일을 로드합니다. 그것은 변환한다
    .mat 파일의 각 프레임을 Frame 클래스 인스턴스에 추가한 다음 "frames"이라는 사전에 포함됩니다.

    Parameters
    ----------
    path : the path where the .mat files are located. (string)

    Returns
    -------
    frames : a dictionary of the form:
             (key:value) -> (<video_name>_<frame_number>:<Frame class instance>)
    """
    frames = {}
    object_gt_files_dict = generate_gt_files_dict(path)

    for key in object_gt_files_dict.keys():
        file_name = object_gt_files_dict[key]

        gt = loadmat(file_name)

        # 프레임 수를 얻으십시오
        frames_number = len(gt["structXML"][0])

        # 각 프레임에 대한 데이터 처리
        for i in range(frames_number):
            image_name = file_name.split("/")[-1].replace("_Haze", "").replace("_ObjectGT.mat", "") + ("_frame%d.jpg" % i)
            image_name = image_name.replace("ObjectGT", "images")
            bb = gt["structXML"][0]["BB"][i]
            objects = gt["structXML"][0]["Object"][i]
            motion = gt["structXML"][0]["Motion"][i]
            distance = gt["structXML"][0]["Distance"][i]
            frame = Frame(i, image_name, bb, objects, motion, distance)
            frames[image_name] = frame
            print("load_mat_files_in_dict[file_name] :" + file_name)
            print("load_mat_files_in_dict[image_name]:" + image_name)

    return frames

def generate_yolo_files(path, output_path):
    """
    지정된 경로의 각 프레임에 대해 YOLO 파일을 생성합니다.

    Parameters
    ----------
    path : the path where the .mat files are located. (string)
    output_path : the desired path for the YOLO files. (string)
    """
    frames = load_mat_files_in_dict(path)
    for frame_key in frames.keys():
        frame = frames[frame_key]
        frame.generate_yolo(output_path)

# 지정된 경로의 각 프레임에 대해 YOLO 파일을 생성하고 원하는 출력 경로에 저장합니다.
for i, file_path in enumerate(PATHS_TO_GT_FILES):
    generate_yolo_files(file_path, OUTPUT_PATHS[i])

# 여러 디렉터리에서 .txt 및 .jpg 파일 수 계산
import os

directory_paths = [
    train_frames_path,
    val_frames_path,
    lab_train_path,
    lab_val_path
]

file_counts = {}
extensions = [".txt", ".jpg"]

for directory_path in directory_paths:
    counts = {}
    for extension in extensions:
        file_count = len([name for name in os.listdir(directory_path) if name.endswith(extension) and os.path.isfile(os.path.join(directory_path, name))])
        counts[extension] = file_count
    file_counts[directory_path] = counts

for directory_path, counts in file_counts.items():
    print(f"Directory: {directory_path}")
    for extension, count in counts.items():
        print(f"Number of {extension} files: {count}")