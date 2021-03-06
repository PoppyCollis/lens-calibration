# lens-calibration
Camera calibration using OpenCV library to correct for lens distortion.

The calibration process involves taking a set of images (you should have at least 15) in which you have a checkerboard of known size. A checkerboard can be printed out and fixed to some kind of flat surface in order to use for calibrating the camera. You should make sure that the checkerboard has a plain border surrounding the pattern at least as large as the width of one of the squares (chessboard-pattern-7x10.png has been provided for you).

1. Open 'camera-calibration.py' in python IDE.
2. Create files in working directory called 'distorted' and 'calibration'.
3. Scroll down to where the main() function is defined and plug in your DV camera server's IP address and port number, and values for nx and ny (these are calculated by adding 1 to the the number of inner columns and rows of your checkerboard respectively), and the path to your images folder (e.g. ./distorted/').
4. Run the script.
5. This should bring up an image window of your DV camera output. To take a photo, press 'space'. This should save each image to the folder distorted. You should change the angle and position of the checkerboard within the image, in order to make sure that your set of input images covers the whole visual field and various different angles within it. When taking each image, make sure to:
  a. include the whole of the checkerboard in the frame (this is including the border surrounding the squares)
  b. set the exposure of the image to an appropriate level for the lighting levels of the room so that there is no glare on the checkerboard
  c. hold the checkerboard close enough to the camera so that a large proportion of the visual field is covered
  d. keep the angles of the checkerboard shallow
  e. take photos that are evenly distributed over parts of the visual field - do not favour one area
When you have finished taking photos (the more the better), press 'q' to proceed to next step.
7. This should bring up a window of an image in which the program has found the checkerboard and drawn the corners. Check to see if the corners have all been drawn in the right place and that none of the checkerboard (or the border) is occluded. To accept the image for calibration press 'space', while pressing any other key will reject the image. It is important that you reject any that don't meet the criteria. Work your way through each image, choosing to accept or reject for calibration (eee example_corners_reject.jpg and example_corners_accept.jpg for examples).
8. The script will then bring up a window of live DV camera output, in which each frame has been undistorted. Explore the visual field with the checkerboard and check the quality of the distortion correction. If unsatisfactory, you may have to go back through and take different / more images.
