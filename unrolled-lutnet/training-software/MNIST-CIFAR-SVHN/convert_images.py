#================================================================
#
#   File name   : convert_images.py
#   Description : Use to convert images from original size to 608*608
#
#================================================================
import cv2
import numpy as np
from PIL import Image

input_size = 608

def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)                 # 608/1080, 608/1920
    nw, nh  = int(scale * w), int(scale * h) # 608, 608/1920*1080=342
    image_resized = cv2.resize(image, (nw, nh)) 

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)          # 608, 608, 3
    dw, dh = (iw - nw) // 2, (ih-nh) // 2                               # 0, 342
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized                  # 
    image_paded = image_paded / 255
    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

label_txt = "mnist/mnist_test.txt"
for line in open(label_txt).readlines():
    image_info = open(label_txt).readlines()[ID].split()
    image_path = image_info[0]
    image_name = image_path.split("/")[-1]
    print(image_name)

original_image      = cv2.imread(image_path)
    # original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    # img = Image.fromarray(image_data)
    # img.save('myimg.jpeg')



#cv2.imwrite('data/dst/lena_bgr_cv.jpg', im_cv)