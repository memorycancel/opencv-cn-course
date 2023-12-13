import os
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
from urllib.request import urlretrieve
matplotlib.rcParams['figure.figsize'] = (9.0, 9.0)

# 下载函数
def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assests....", end="")
    urlretrieve(url, save_path)
    try:
        with ZipFile(save_path) as z:
            z.extractall(os.path.split(save_path)[0])
        print("Done")
    except Exception as e:
        print("\nInvalid file.", e)

# 下载
URL = r"https://www.dropbox.com/s/48hboi1m4crv1tl/opencv_bootcamp_assets_NB3.zip?dl=1"
asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_NB3.zip")
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)

image = cv2.imread("Apollo_11_Launch.jpg", cv2.IMREAD_COLOR)

# 展示原始示例图片“阿波罗11号”
plt.imshow(image[:, :, ::-1])