import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

images = glob.glob('camera_cal/calibration*.jpg')

# prepare object points
nx = 9
ny = 6

objpoints = [] # 3D points on distorted images
imgpoints = [] # 2D points on undistorted image

objp = np.zeros((nx*ny, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Make a list of calibration images
for fname in images:
    img = cv2.imread(fname)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    print(ret)
    # If found, add object points and image points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

_, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "calibration_pickle.p", "wb" ) )
# test undistort
cal_pickle = pickle.load( open( "calibration_pickle.p", "rb" ) )
m = cal_pickle["mtx"]
d = cal_pickle["dist"]
test_img = cv2.imread('camera_cal/calibration1.jpg')
undistorted = cv2.undistort(test_img, m, d, None, m)
plt.imshow(undistorted)
plt.show()

