#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jun 14 12:40:18 2021

@author: poppy
see OpenCV documentation: https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
"""

from glob import glob
import numpy as np
import cv2
import dv
import PIL
from PIL import Image
import itertools
from itertools import product
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


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
    #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    
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
    np.save("./point-wise-mapping/disorted.npy", image)
    
    #add circle...
    
    img = cv2.circle(image,(320,20), 10, (0,0,255), -1)
    cv2.imshow('circle', img)
    

    image = cv2.undistort(image, mtx, dist, None, newcameramtx)
    cv2.imshow('./undisorted.npy', image)
    np.save("./point-wise-mapping/udisorted.npy", image)
    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()

#add in point (can add more than one) to find undistorted point
#2xN/Nx2 1-channel or 1xN/Nx1 2-channel (CV_32FC2 or CV_64FC2) (or vector<Point2f> ).
def undistort_point(mtx, dist):
    distorted = np.array([ [ [10,340], [125,10], [10,10], [250,10], [10,175], [125, 175], [250, 340], [125, 340], [250, 175], [65,250], [195,250], [65, 90], [195, 90] ] ], dtype="float64")
    print(distorted)
    
    x, y = distorted.T
    plt.xlim(0,261)
    plt.ylim(0,347)
    plt.scatter(x,y)
    plt.show()
    
    
    print(distorted.shape)
    undistorted = cv2.undistortPoints(distorted, mtx, dist)
    print(undistorted)
    print(undistorted.shape)
    
    undist = np.add(distorted, undistorted)
    
    
    x, y = undist.T
    plt.xlim(0,261)
    plt.ylim(0,347)
    plt.scatter(x,y)
    plt.show()
    
def main():
    #dv_address = '192.168.1.100'
    dv_address = '127.0.0.1'
    dv_frame_port = 36001 
    nx  = 6 # This is calculated by adding 1 to the the number of inner columns of your checkerboard
    ny = 9 # This is calculated by adding 1 to the the number of inner rows of your checkerboard
    f_dir = './distorted/'
    get_dv_image(dv_address, dv_frame_port, f_dir)
    mtx, dist, newcameramtx = find_calib_parameters (nx, ny, f_dir)
    test_undistort(dv_address, dv_frame_port, mtx, dist, newcameramtx)
    undistort_test_image(mtx, dist, newcameramtx)
    undistort_point(mtx, dist)
 
"""
    #convert numpy array into x, y coordinates
    
    image = np.load("./distorted/1.npy")
    arr = image
    
    arr = np.array([
        list(range(347))
        for _ in range(261)
    ])
    
    
    print(arr.shape)
    # (200, 300)
    
    pixels = arr.reshape(-1)
    
        #n-dimension solution
        #coords = map(range, arr.shape)
        #indices = np.array(list( product(*coords) ))
    
    xs = range(arr.shape[0])
    ys = range(arr.shape[1])
    indices = np.array(list(product(xs, ys)))
    
    
    pd.options.display.max_rows = 20
    
    index = pd.Series(pixels, name="pixels")
    df = pd.DataFrame({
        "x" : indices[:, 0],
        "y" : indices[:, 1]
    }, index=index)
    print(df)
    
    xy_coord = df.to_numpy()
    
    
    print(xy_coord) #array of x, y coordinates, change to dtype="float64"
    
    xy_coord = xy_coord.astype(np.float64)
    
    #undistort each x,y point and put into a new array 
    
    undistorted = cv2.undistortPoints(xy_coord, mtx, dist)
    print(undistorted)
    
    """

if __name__ == "__main__":
    main()
    
    
    
 
    """  
    image = np.load("./distorted/1.npy")
    print(a.shape)
    print(a)
    for x,y in image:
            distorted = np.array([ [ [x,y] ] ], dtype="float64")
            undistorted = cv2.undistortPoints(distorted, mtx, dist)
    
    point_matrix = np.zeros(shape = (261,347,2) dtype="float64")
    point_matrix.append

        
    print(distorted)
    undistorted = cv2.undistortPoints(distorted, mtx, dist)
    print(undistorted)
    
    
    
    point_matrix = np.zeros(shape = (261,347,2) dtype="float64")
    distorted = np.array([ ???], dtype="float64")
    for x,y in distorted:
        
        
    np.argwhere((image ==[255,0,0]).all(axis=2))
""" 
