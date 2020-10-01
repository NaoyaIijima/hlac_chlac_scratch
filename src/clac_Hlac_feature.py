import cv2
import utils


img = cv2.imread("./../data/test.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

hlac_feature = utils.calc_hlac_dev(img)
print("hlac_feature:", hlac_feature)
