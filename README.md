# Quick Start Guide

## 1. Command
1. 將拍好的照片放到 ./cameraAll/kinect/
2. 執行下方command
3. ./camera_matrix/kinect 中各相機 yml 的 Extrinsic 即為所要。 (目前只有 Extrinsic 可以轉 OpenGL 格式)
```python=
    python detection.py -i ./cameraAll/kinect/ -o ./camera_matrix/kinect -t DICT_5X5_100 -a setting.json -res 1920x1080 -thres 100 -opengl true
```
-i : path to input image containing ArUCo tag  
-o : path to output calibrated camera matrix    
-t : type of ArUCo tag to detect  
-a : json of aruco coordinate  
-res : calibration image resolution  
-thres : the black and white threshold of binary image (0~255)  
-vis: visualize binary image result (Default is False)  
-opengl : use OpenGL coordinate system (Default is OpenCV)

## 2.  OpenCV 與 OpenGL 座標系
OpenCV 與 OpenGL 都是右手座標系. OpenCV 座標系 只要沿著 x 軸 旋轉 180 度 就是 OpenGL 座標系.

OpenCV 座標系：
```
z
-----> x
|
|
v
y
```
OpenGL 座標系：
```
y
^
|
|
-----> x
z
```
colmap 的坐标系和 OpenCV 一致，blender 坐标系和 OpenGL 一致。

OpenCV 是 w2c, camera = R * world + T

OpenGV 是 c2w, world = R * camera + T

例: 將OpenCV轉OpenGV
```
extrinsic = np.linalg.inv(extrinsic)
extrinsic = np.array([extrinsic[0], extrinsic[2], extrinsic[1], extrinsic[3]])
```


