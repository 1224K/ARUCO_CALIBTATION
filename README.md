# Quick Start Guide

## 1. Command

```python=
    python detection.py -i .\cameraAll\kinect\ -o ./camera_matrix/kinect -t DICT_5X5_100 -a setting.json -res 1920x1080 -thres 100 -opengl true
```
-i : path to input image containing ArUCo tag  
-o : path to output calibrated camera matrix    
-t : type of ArUCo tag to detect  
-a : json of aruco coordinate  
-res : calibration image resolution  
-thres : the black and white threshold of binary image (0~255)  
-vis: visualize binary image result (Default is False)
-opengl : use OpenGL coordinate system (Default is OpenCV)

## 2.  