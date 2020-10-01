import cv2
import sys
import numpy as np
import pandas as pd
import utils


video_path = "./../data/test.mp4"
cap = cv2.VideoCapture(video_path)
FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

images = []

# add a black image beginning of the video
images.append(np.zeros([height + 2, width + 2]))

for i in range(FRAMES):
    _, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # add a black pixel around the image
    black_img = np.zeros([height + 2, width + 2])
    black_img[1:-1, 1:-1] = img

    images.append(black_img)

# add a black image ending of the video
images.append(np.zeros([height + 2, width + 2]))
images = np.array(images).astype(np.int32)

cap.release()

mask_filepath = "./../config/mask_chlac.csv"
masks, mask_n = utils.prepare_masks_chlac(mask_filepath)

chlac_features = []
for f in range(FRAMES):
    sys.stdout.write("\r" + str(round(100 * f / FRAMES, 1)) + "%")
    sys.stdout.flush()

    # extract images for 3 frames to create voxel data
    data = images[f: f + 3, :, :]

    div_data = utils.split2boxel(data)
    # div_data = utils.split2boxel_(data)  # old version, too slow

    feature = utils.calc_chlac_dev(div_data, masks, mask_n)
    # feature_ = utils.calc_chlac(
    #     div_data, masks, mask_n)  # old version, too slow
    # if np.array_equal(feature, feature_):
    #     print("features of the two method are equel.")

    chlac_features.append(feature)

# not use chlac feature of initial and last frame
chlac_features = chlac_features[1:-1]

chlac_features = pd.DataFrame(np.array(chlac_features))
chlac_features.to_csv("./../results/chlac_features.csv",
                      index=None, header=None)

print()
print("end")
