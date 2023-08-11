import pathlib
import cv2


def get_image(image_file: str or pathlib.Path, flag=cv2.IMREAD_UNCHANGED):
    img = cv2.imread(str(image_file), flag)
    return img
