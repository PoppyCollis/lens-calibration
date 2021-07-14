#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jun 14 12:40:18 2021

@author: poppy
"""

from glob import glob
import numpy as np
import cv2
import dv
import PIL
from PIL import Image

# Create files in working directory called "distorted" and "calibration"


def get_dv_image(dv_address, dv_frame_port, f_dir):
    with dv.NetworkFrameInput(dv_address, dv_frame_port) as dv_frame_f:
        i_frame = 0        
        for frame in dv_frame_f:
            cv2.imshow('frame', frame.image)
            k = cv2.waitKey(1)
            if k == ord(' '):
                print('image', i_frame)
                f_path = f_dir + str(i_frame) + '.npy'
                np.save(f_path, frame.image)
                i_frame += 1
            elif k == ord('q'):
                break
            

def find_calib_parameters(nx, ny, f_dir='./'):
    # Termination criteria allows you to specify the desired accuracy or change in parameters 
    #at which the iterative algorithm stops (EPS) and the maximum number of iterations or 
    #elements to compute (MAX_ITER)

    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    # Prepare object points, like (0, 0, 0), (1, 0, 0), (2, 0, 0) ..., (6, 5, 0)
    objp = np.zeros((nx * ny, 3), np.float32) 
    objp[:, :2] = np.mgrid[0:ny, 0:nx].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3D point in real world space
    imgpoints = [] # 2D points in image plane.
    image_files = glob(f_dir + '*.npy')

    for f_name in image_files:
        image = np.load(f_name)
        # cvtColor converts an image from one color space to another (in this case to greyscale)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

        # The function attempts to determine whether the input image is a view of the 
        #chessboard pattern and locate the internal chessboard corners. The function 
        #returns a non-zero value if all of the corners are found and they are placed 
        #in a certain order (row by row, left to right in every row). Otherwise, if 
        #the function fails to find all the corners or reorder them, it returns 0. 
        ret, corners = cv2.findChessboardCorners(gray, (ny, nx), None)

        # If found, add object points, image points (after refining them using SubPix method)
        if ret == True:
            print(f_name)
            
            # This conducts a more refined search in a smaller area of pixels in an attempt to find the corners with greater (subpixel) accuracy
            #corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

            # The function draws individual chessboard corners detected either as red circles if the board was not found, 
            #or as colored corners connected with lines if the board was found.
            cv2.drawChessboardCorners(image, (ny, nx), corners, ret)
            cv2.imshow('image', image)
            k = cv2.waitKey(0)

            if k == ord(' '):
                objpoints.append(objp)
                imgpoints.append(corners)
                print(f_name)
                
            elif k == ord('q'):
                break
            
    cv2.destroyAllWindows()        
    
    # Record and save the calibration parameters
    # The function estimates the intrinsic camera parameters and extrinsic parameters for each of the views.
    # The coordinates of 3D object points and their corresponding 2D projections in each view must be specified.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    #image = cv2.imread('./test-images/xtr_test_5.jpg')
    image = np.load('./distorted/0.npy')
    h,  w = image.shape[:2]
    # Returns the new camera intrinsic matrix based on the distortion coefficients
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    np.save("ncmtx.npy", newcameramtx)
    # Make sure to create a file of this name in your current directory for you arrays to be saved to

    np_mtx = np.array(mtx)
    np_dist = np.array(dist)
    np_rvecs = np.array(rvecs)
    np_tvecs = np.array(tvecs)
    
    np.save("./calibration/mtx.npy", np_mtx)   
    np.save("./calibration/dist.npy", np_dist)
    np.save("./calibration/rvecs.npy", np_rvecs)
    np.save("./calibration/tvecs.npy", np_tvecs)     
        
    cv2.destroyAllWindows()

    return mtx, dist, newcameramtx     
    
def test_undistort(dv_address, dv_frame_port, mtx, dist, newcameramtx):
    # Connect to DV camera

    with dv.NetworkFrameInput(dv_address, dv_frame_port) as dv_frame_f:
        print('connected')
        
        # Undistorts video output of camra frame by frame
        for frame in dv_frame_f:
            image = frame.image
            # Function transforms an image to compensate for lens distortion                
            cv2.imshow('distorted', image)
            image = cv2.undistort(image, mtx, dist, None, newcameramtx)
            
            #image = cv2.undistort(image, mtx, dist)
            
            cv2.imshow('undistorted', image)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break

    cv2.destroyAllWindows()            

#comapre the lens distortion using just a single image
def undistort_test_image(mtx, dist, newcameramtx):
    image = np.load("./distorted/1.npy")
    cv2.imshow('distorted', image)
    image = cv2.undistort(image, mtx, dist, None, newcameramtx)
    cv2.imshow('./comparison/undistorted', image)
    np.save("test_1.npy", image)
    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()


def main():
    dv_address = '192.168.1.100'
    dv_frame_port = 36001 
    nx  = 6 # This is calculated by adding 1 to the the number of inner columns of your checkerboard
    ny = 9 # This is calculated by adding 1 to the the number of inner rows of your checkerboard
    f_dir = './distorted/'
    get_dv_image(dv_address, dv_frame_port, f_dir)
    mtx, dist, newcameramtx = find_calib_parameters (nx, ny, f_dir)
    test_undistort(dv_address, dv_frame_port, mtx, dist, newcameramtx)
    undistort_test_image(mtx, dist, newcameramtx)

if __name__ == "__main__":
    main()
    



# undistort
#dst = cv2.undistort(image, mtx, dist, None, newcameramtx)

# crop the image
#x, y, w, h = roi
#dst = dst[y:y + h, x:x + w]

#cv2.imwrite('calibrate_result.png', dst)

#cv2.imshow('calibrate result', dst)
#cv2.waitKey(0)
