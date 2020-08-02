from skimage import feature
import numpy as np
import cv2
class local_binary_pattern:
	def __init__(self, num_points, radius):
		self.numPoints = num_points
		self.radius = radius

	def get_feature(self, image, eps=1e-7):
		lbp = feature.local_binary_pattern(image, self.numPoints,self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, self.numPoints + 3),range=(0, self.numPoints + 2))
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		return hist


def sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    kp, desc = sift.detectAndCompute(gray_img, None)
    desc = desc[:20, :]
    output = desc.flatten()
    #print(output.shape)
    return output
