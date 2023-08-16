import pathlib
import numpy as np
import cv2


def get_image(image_file: str or pathlib.Path, to_gray: bool = False, flag=cv2.IMREAD_UNCHANGED):
    img = cv2.imread(str(image_file), flag)
    try:
        if to_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, -1)  # Add channels dim
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as err:
        print(err)
    return img
