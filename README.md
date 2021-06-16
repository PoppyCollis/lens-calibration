# lens-calibration
Camera calibration using opencv library to correcting for lens distortion

camera-calibration.py involves taking a set of images (you should have at least 15), in which you have a checkerboard of known size. A checkerboard can be printed out and fixed to some kind of flat surface in order to use for taking images. You should make sure that the checkerboard has a plain border surrounding the pattern at least as large as the width of one of the squares (chessboard-pattern-7x10.png has been provided for you). The whole border should be visible in each of the images. In each of the instance, you should change the angle and position of the checkerboard within the image, in order to make sure that your set of input images covers the whole visual field and various different angles within it.

get-dv-image.py file includes the process of taking a photo using the dynamic vision sensor camera
