import os
import argparse
import string
import sys
import numpy as np
from utils import aruco_util as au
from utils import file_util as fu
from threed import reconstruction as rc
import cv2
import math

# Camera Named Rule
cam_names = ["camera0.txt", "camera1.txt", "camera2.txt", "camera3.txt", "camera4.txt"]
# Camera Named Rule
image_names = ["camera0.png", "camera1.png", "camera2.png", "camera3.png", "camera4.png"]


''' GoPro Paramters '''
linear_off = [92, 61] # hfov, vfov
narrow_off = [73, 45] # hfov, vfov

# Get Intrinsic, Extrinsic, Projection Matrix
def GetMatrix(floder, cam):
    file_path = os.path.join(floder, cam)

    # Define 4X4 matrix
    intrinsic_matrix = np.zeros((4,4),dtype=float)
    extrinsic_matrix = np.zeros((4,4),dtype=float)  
    projection_matrix = np.zeros((4,4),dtype=float)  
    
    f = open(file_path)
    lines = f.readlines()
    
    row = 0
    for line in lines:
        list = line.strip('\n').replace('\t', ' ').split(' ')
        if row < 4:
            intrinsic_matrix[row:] = list[0:4]
        elif row < 8:
            extrinsic_matrix[row-4:] = list[0:4]
        elif row < 12:
            projection_matrix[row-8:] = list[0:4]
        
        row += 1
        if row==12: 
            break
    
    # Show Matrix
    # print("Camera: ", cam.split('.')[0], " Infomation")
    # print("Intrinsic Matrixis :")
    # print(intrinsic_matrix)
    # print("Extrinsic Matrix is :")  
    # print(extrinsic_matrix)
    # print("Projection Matrix is :")  
    # print(projection_matrix)
    # print("=====================================================")
    # print("")

    return intrinsic_matrix, extrinsic_matrix, projection_matrix


# Count cameras
def GetCameras(dir_name : string):
    temp = os.listdir(dir_name)
    cam_list = []
    for names in temp:
        if names.endswith(".png"):
            cam_list.append(names)

    print(cam_list)
    return cam_list


''' The return value is the intrinsic matrix of each camera '''
def camera_intrinsic_transform(vfov=45,hfov=73,pixel_width=1920,pixel_height=1080):
    camera_intrinsics = np.zeros((3,3))
    camera_intrinsics[2,2] = 1
    camera_intrinsics[0,0] = (pixel_width/2.0)/math.tan(math.radians(hfov/2.0))
    camera_intrinsics[0,2] = pixel_width/2.0
    camera_intrinsics[1,1] = (pixel_height/2.0)/math.tan(math.radians(vfov/2.0))
    camera_intrinsics[1,2] = pixel_height/2.0
    return camera_intrinsics

def get_img_resolution(arg_res : str):
    width, height = arg_res.split('x')
    return [int(width), int(height)]

def main(args):
    """Get aruco position"""
    aruco_position_path = args["aruco_pos"]
    dict_points =  fu.read_aruco_json(aruco_position_path)

    """ Read cameras & Calculate Intrinsic Matrix """
    res = get_img_resolution(args['resolution'])
    print(res[0], res[1])
    
    cameras = GetCameras(args["image"]) # return a camera list

    # camera_intrinsic = camera_intrinsic_transform(linear_off[1], linear_off[0], res[0], res[1]) # get the instrinsic matrix
    # camera_intrinsic = camera_intrinsic_transform(59, 90, res[0], res[1]) # get the instrinsic matrix
    # camera_intrinsics = np.zeros((3,3))
    # camera_intrinsic = np.array([[600.2734275, 0, 642.26062], [0, 600.18829, 365.644], [0, 0, 1]])
    # camera_intrinsic = np.array([[600.18829, 0, 365.644], [0, 600.2734275, 642.26062], [0, 0, 1]])
    camera_intrinsic = np.array([[900.41015625, 0, 963.6409301757812], [0, 900.282470703125, 548.7173461914062], [0, 0, 1]])
    # print(camera_intrinsic)

    """ Detect ArUcos & Calculate Extrinsic Matrix """
    binary_threshold = args["threshold"]
    
    for i, cam in enumerate(cameras):
        # print("")
        # au.detectAruco(os.path.join(args["image"], image_names[i]), args["type"], camera_Matrix[i], cam)
        rvecs, tvecs, extrinsic, camera_matrix, distortion_coeffs = au.detectAruco(args["output"], os.path.join(args["image"], cameras[i]), args["type"], camera_intrinsic, cam, dict_points, res, binary_threshold, args["visualize"])
        # au.DrawResult(cv2.imread(os.path.join(args["image"]), camera_matrix, distortion_coeffs, rvecs, tvecs, cam, markerID)
        ### Store images

        ### Write Matrice
        ### Required: Rotation vectors, translation vectors, distortion coeffs, extrinsic matrix and camera matirx
        file_dir = os.path.join(args["output"])
        if args["OpenGL"]:
            extrinsic = np.linalg.inv(extrinsic)
            extrinsic = np.array([extrinsic[0], extrinsic[2], extrinsic[1], extrinsic[3]])
        fu.writeYAML(cam, rvecs.reshape(1,3), tvecs.reshape(1,3), extrinsic, camera_matrix, file_dir)
        # fu.writeJson(cam, rvecs.reshape(1,3), tvecs.reshape(1,3), extrinsic, camera_matrix, file_dir)
        # fu.writeXml(cam, distortion_coeffs, extrinsic, camera_matrix, file_dir)
        # fu.writeDat(cam, extrinsic, camera_matrix, file_dir)

    return
### --------------------------------------------------------modify by K-----------------------------------------------
    file_dir = os.path.join(args["output"])
    from os.path import join
    print("----------------------------------")
    intri_name = join(file_dir, 'intri.yml')
    extri_name = join(file_dir, 'extri.yml')
    intri = FileStorage(intri_name, True)
    extri = FileStorage(extri_name, True)
    results = {}
    cameras.sort()
    camnames = [key_.split('.')[0] for key_ in cameras]

    print(camnames)
    intri.write('names', camnames, 'list')
    extri.write('names', camnames, 'list')
    for key_ in cameras:
        key = key_.split('.')[0]
        # print(key)
        rvecs, tvecs, extrinsic, camera_matrix, distortion_coeffs = au.detectAruco(args["output"], os.path.join(args["image"], key_), args["type"], camera_intrinsic, key_, dict_points, res, binary_threshold, args["visualize"])
        intri.write('K_{}'.format(key), camera_matrix)
        intri.write('dist_{}'.format(key), distortion_coeffs[:5]) ##TODO: check distortion_coeffs
        extri.write('R_{}'.format(key), rvecs)
        extri.write('Rot_{}'.format(key), extrinsic[0:3,0:3])
        extri.write('T_{}'.format(key), tvecs)
### --------------------------------------------------------modify by K-----------------------------------------------

### --------------------------------------------------------modify by K-----------------------------------------------
class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = open(filename, 'w')
            self.fs.write('%YAML:1.0\r\n')
            self.fs.write('---\r\n')
        else:
            assert os.path.exists(filename), filename
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.isWrite = isWrite

    def __del__(self):
        if self.isWrite:
            self.fs.close()
        else:
            cv2.FileStorage.release(self.fs)

    def _write(self, out):
        self.fs.write(out+'\r\n')

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            self._write('{}: !!opencv-matrix'.format(key))
            self._write('  rows: {}'.format(value.shape[0]))
            self._write('  cols: {}'.format(value.shape[1]))
            self._write('  dt: d')
            self._write('  data: [{}]'.format(', '.join(['{:.6f}'.format(i) for i in value.reshape(-1)])))
        elif dt == 'list':
            self._write('{}:'.format(key))
            for elem in value:
                self._write('  - "{}"'.format(elem))
        elif dt == 'int':
            self._write('{}: {}'.format(key, value))

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        elif dt == 'int':
            output = int(self.fs.getNode(key).real())
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)
### --------------------------------------------------------modify by K-----------------------------------------------


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True,
	                help="path to input image containing ArUCo tag")
    ap.add_argument("-o", "--output", type=str, required=True,
	                help="path to output calibrated camera matrix")
    ap.add_argument("-t", "--type", type=str, required=True,
	                default="DICT_ARUCO_ORIGINAL",
	                help="type of ArUCo tag to detect")
    ap.add_argument("--camera_matrix_path", type=str,
	                default="matrix",
	                help="origin camera matrix")
    ap.add_argument("--world_setting_json", type=str,
                    default=".",
                    help="json record world information, ex: aruco coordinate")
    ap.add_argument("-a", "--aruco_pos", type=str,
                    default="aruco_pos.json",
                    help="json of aruco coordinate")
    ap.add_argument("-res", "--resolution", type=str,
                    default="1920x1080",
                    help="calibration image resolution")
    ap.add_argument("-thres", "--threshold", type=int,
                    default="the black and white threshold of binary image")
    ap.add_argument("-vis", "--visualize", type=bool, default=False, help="Display binary threshold result")
    ap.add_argument("-opengl", "--OpenGL", type=bool, default=False, help="OpenGL coordinate system")
    args = vars(ap.parse_args())

    main(args)
    
# python detection.py -i ./cameraAll/Example -o ./camera_matrix/Example4 -t DICT_5X5_100 -a setting.json -res 1920x1080 -thres 100