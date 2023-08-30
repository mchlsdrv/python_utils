import pathlib
import numpy as np
import cv2
import torch


def get_image(image_file: str or pathlib.Path, to_gray: bool = False, to_tensor: bool = False,
              channel_first: bool = False, add_channel_dim: bool = True, add_batch_dim: bool = False,
              device: torch.device = torch.device('cpu'), flag=cv2.IMREAD_UNCHANGED):
    img = cv2.imread(str(image_file), flag)
    try:
        if to_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if add_channel_dim:
                img = np.expand_dims(img, -1)  # Add channels dim

        # - Convert to RGB from BGR of cv2
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if to_tensor:

            # - Change from HWC -> CHW
            if channel_first:
                img = np.transpose(img, (2, 0, 1))

            # - Add the batch dim
            if add_batch_dim:
                img = np.expand_dims(img, 0)

            # - Convert to tensor
            img = torch.tensor(img, dtype=torch.float)

            # - Move to device
            img = img.to(device)

    except Exception as err:
        print(err)
    return img
