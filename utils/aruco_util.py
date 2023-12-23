from msilib import make_id
import cv2
from cv2 import solvePnP
import imutils
import numpy as np
import os
import sys
import math
sys.path.append(os.path.join("D:/Calibration/", "utils"))
from utils import file_util as fu

split_line = "=========="
aruco_dir = "aruco_detection"
aruco_dir = os.path.join(aruco_dir, "Example")
aruco_ids = ["42","56","66","79","87"]
### Corners
corner_42 = None
corner_56 = None
corner_66 = None
corner_79 = None
corner_87 = None

realpos_42 = np.array([[-0.5, 0.0, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.5, 0.0, -0.5],
                    [-0.5, 0.0, -0.5]])

realpos_42_simulate = np.array([[-0.09, -0.8, 0.09],
                [0.09, -0.8, 0.09],
                [0.09, -0.8, -0.09],
                [-0.09, -0.8, -0.09]])
# dict_points = {"42": realpos_42,
#                 "56": realpos_42 + np.array([-0.8, 0, -0.3]),
#                 "66": realpos_42 + np.array([0, 0, 1]),
#                 "79": realpos_42 + np.array([1.24, 0, 0.328]), 
#                 "87": realpos_42 + np.array([0, 0, -1.5])
#             }
"""Debug"""
# dict_points = {"42": realpos_42,
#                 "56": realpos_42 + np.array([0.845, 0, -1.19]),
#                 "66": realpos_42 + np.array([-1.203, 0, -0.88]),
#                 "79": realpos_42 + np.array([1.175, 0, 0.83]), 
#                 "87": realpos_42 + np.array([-1.165, 0, 1.12])
#             }
"""Debug"""
# dict_points = {"42": realpos_42,
#                 "56": realpos_42 + np.array([0.845, 0, -1.19]),
#                 "66": realpos_42 + np.array([-1.203, 0, -0.88]),
#                 "79": realpos_42 + np.array([1.175, 0, 0.83]), 
#                 "87": realpos_42 + np.array([-1.165, 0, 1.12])
#             }

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
#	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
#	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
#	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
#	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def setMarkerInfo(marker_id, markerCorner):
    corner = markerCorner.reshape((4, 2))

    if marker_id == 42:
        corner_42 = corner
    elif marker_id == 56:
        corner_56 = corner
    elif marker_id == 66:
        corner_66 = corner
    elif marker_id == 79:
        corner_79 = corner
    elif marker_id == 87:
        corner_87 = corner
    
    return corner

# def DrawMarker():
def detectAruco(dir, img_path, aruco_type, camera_matrix, cam, dict_points, res : list, aruco_threshold=210, visualize=False):

    print(split_line + " Start Detect Aruco " + split_line)
    
    # aruco_dir = os.path.join(aruco_dir, "221018_test")

    # load the input image from disk and resize it
    print("[Stage1] loading image...")
    ori_image = cv2.imread(img_path)

    grayscale_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)
    ret, binary_image = cv2.threshold(grayscale_image, aruco_threshold, 255, cv2.THRESH_BINARY) # convert rgb image to binary
    
    image = binary_image
    image = cv2.resize(image, (res[0], res[1]))
    if(visualize):
        cv2.waitKey(0)
        # cv2.imshow("Resized", image)
        cv2.imshow("Axes", image)
        cv2.waitKey(0) 

    # verify that the supplied ArUCo tag exists and is supported by
    # OpenCV
    if ARUCO_DICT.get(aruco_type, None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(aruco_type))
        return 

    # load the ArUCo dictionary, grab the ArUCo parameters, and detect
    # the markers
    print("[INFO] detecting '{}' tags...".format(aruco_type))
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
    arucoParams = cv2.aruco.DetectorParameters_create()
    # arucoParams.minDistanceToBorder =  7
    # arucoParams.cornerRefinementMaxIterations = 149
    # arucoParams.minOtsuStdDev= 4.0
    # arucoParams.adaptiveThreshWinSizeMin= 7
    # arucoParams.adaptiveThreshWinSizeStep= 49
    arucoParams.minMarkerDistanceRate= 0.014971725679291437
    # arucoParams.maxMarkerPerimeterRate= 10.075976700411534 
    # arucoParams.minMarkerPerimeterRate= 0.2524866841549599 
    # arucoParams.polygonalApproxAccuracyRate= 0.05562707541937206
    # arucoParams.cornerRefinementWinSize= 9
    # arucoParams.adaptiveThreshConstant= 9.0
    # arucoParams.adaptiveThreshWinSizeMax= 369
    # arucoParams.minCornerDistanceRate= 0.09167132584946237
    # arucoParams.aprilTagMinWhiteBlackDiff = 200
    # arucoParams.aprilTagQuadSigma = 0.8
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    print(len(corners))
    print(corners)
    print(rejected)
    
    dict_corners = {}
    
    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()

        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            
            corner = setMarkerInfo(markerID, markerCorner)
            
            dict_corners[str(markerID)] = corner
            (topLeft, topRight, bottomRight, bottomLeft) = corner

            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            cv2.line(ori_image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(ori_image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(ori_image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(ori_image, bottomLeft, topLeft, (0, 255, 0), 2)

            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(ori_image, (cX, cY), 4, (0, 0, 255), -1)
            cv2.putText(ori_image, '(' + str(cX) + ', ' + str(cY) + ')', (cX, cY - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # draw the ArUco marker ID on the image
            cv2.putText(ori_image, str(markerID),
                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            print("[INFO] ArUco marker ID: {}".format(markerID))

            # show the output image
            # cv2.imshow("Image", image)

            # store the images with aruco
            aruco_cam_dir = os.path.join(aruco_dir, cam.split('.')[0])

            try:
                os.makedirs(aruco_cam_dir)
            except FileExistsError:
                print("資料夾已存在。")
            
            file_name = str(markerID) +'.png'
            save_path = os.path.join(aruco_cam_dir, file_name)
            print(save_path)
            # cv2.imshow("Axes", image)
            cv2.imwrite(save_path, ori_image)
            cv2.waitKey(0)

    # # Arcuo real size
    # print(dict_corners.keys())

    model_points = []
    all_corners = []

    ## Find marker 
    find_marker = False
    for aruco in aruco_ids:
        for key, value in dict_corners.items():
            if key == aruco:
                print("Detect aruco id: " + aruco)
                all_corners.append(value)
                model_points.append(dict_points[key])
                find_marker = True
        
        # If find marker, exit
        # if find_marker:
        #     break
    
    model_points = np.array(model_points).reshape(-1,3)
    all_corners = np.array(all_corners).reshape(-1,2)
    # print(model_points)
    # print(all_corners)
                
    # model_point = np.array(model_point, dtype=np.float64)
    # # model_point[:,0] /= 2.0
    # # model_point[:,2] /= 2.0
    # corner_42 = np.array(corner_42)
    
    # # model_point = model_point.reshape((12,3))
    # # corner_42 = corner_42.reshape((12,2))
    
    # print(model_point)
    distortion_coeffs = np.zeros((14,1))
    # distortion_coeffs = np.array([1.0690112558567140e+01, -9.6336874018415230e+01,
    # 2.0264798739414195e-03, -3.8447954720187108e-03,
    # 2.5169330378722270e+02, 1.0700809869189818e+01,
    # -9.6443441269775533e+01, 2.5207806824596977e+02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # distortion_coeffs = np.array([124.5191719471384, 1918.5765309006081,
    #                                 -0.014827664006143108, -0.0073500210483379508,
    #                                 781.02123041864843, 125.06656360875164, 
    #                                 1898.0506728254268, 1108.9125078663619,
    #                                 0., 0., 0., 0., 0., 0.])
    # camera_matrix = np.array([[-613.434151034, 0.0, 1280.0/2],
    #                   [0.0,  -552.900317856, 720.0/2],
    #                   [0.0, 0.0, 1]])
    # camera_matrix = np.array([[1886.77874518, 0.0, 1920.0/2],
    #                   [0.0,  1833.47616887, 1080.0/2],
    #                   [0.0, 0.0, 1]])
    """ Test 1"""
    # camera_matrix = np.array([[1221.5544090729036, 0.0, 960.0],
    #                   [0.0,  1236.0235057404111, 540],
    #                   [0.0, 0.0, 1]])
    """ Test 2"""
    # camera_matrix = np.array([[852.60923620091432, 0.0, 928.60419711651434],
    #                   [0.0,  852.10248990850516, 531.24191694982630],
    #                   [0.0, 0.0, 1]])
    """Test 3"""
    # camera_matrix = np.array([[9.0898032741007989e+02, 0.0, 9.4638012307875215e+02],
    #                   [0.0,  9.0891595483127401e+02, 5.4046415004394885e+02],
    #                   [0.0, 0.0, 1]])
    if len(corners) <= 0:
        return [], [], [], [], []
    sucess, rvecs, tvecs = cv2.solvePnP(np.array(model_points), np.array(all_corners), camera_matrix, distortion_coeffs)
    # sucess, rvecs, tvecs, inliers = cv2.solvePnPRansac(np.array(model_points), np.array(all_corners), camera_matrix, distortion_coeffs)
    # print(inliers)
    # DrawReprojection(dir, cam, ori_image, camera_matrix, distortion_coeffs, rvecs, tvecs, model_points)
    # _r, _t, _= cv2.aruco.estimatePoseSingleMarkers([all_corners], 1, camera_matrix, distortion_coeffs)
    # print(f'R: {_r}')
    # print(f'T: {_t}')
    # print(f'O: {_}')
    # _aruco_axes = cv2.drawFrameAxes(ori_image, camera_matrix, distortion_coeffs, rvecs, tvecs, 40)
    # cv2.imshow("Axes", _aruco_axes)
    # aruco_axes = cv2.drawFrameAxes(ori_image, camera_matrix, distortion_coeffs, _r, _t, 40)
    # cv2.imshow("Axes", aruco_axes)

    if sucess:
        # print(rvecs)
        # print(rvecs * 180.0 / math.pi)
        # print(tvecs)
        pass
    else:
        print("[Error] Can't generate extrinsic parameters")

    # default intrinsic:
    #     [935.3074, 0.0     , 960.0]
    #     [0.0     , 935.3074, 540.0]
    #     [0.0     , 0.0     , 1.0  ]
    # print(type(camera_matrix))

    # Rvec, tvec to extrinsic matrix
    extrinsic = np.eye(4,4)
    R = cv2.Rodrigues(rvecs.reshape(1,3))
    # print(np.linalg.inv(R[0]))
    extrinsic[0:3,0:3] = R[0]
    extrinsic[0:3,3] = tvecs.reshape(1,3)

    # Draw axis and store the images with aruco
    # DrawResult(dir, image, camera_matrix, distortion_coeffs, rvecs, tvecs, cam, markerID)

    return rvecs, tvecs, extrinsic, camera_matrix, distortion_coeffs

def DrawReprojection(dir, cam, image, camera_matrix, distortion_coeffs, rvecs, tvecs, model_points):
    repro, jaba = cv2.projectPoints(model_points, rvecs, tvecs, camera_matrix, distortion_coeffs)
    repro = np.array(repro, dtype=int).reshape((int)(repro.shape[0]/4), 4, 2)
    print(f'Reprojection: {repro}')
    
    for points in repro:
        # draw the bounding box of reprojection
        cv2.line(image, points[0], points[1], (255, 0, 0), 2)
        cv2.line(image, points[1], points[2], (255, 0, 0), 2)
        cv2.line(image, points[2], points[3], (255, 0, 0), 2)
        cv2.line(image, points[3], points[0], (255, 0, 0), 2)
    camera_name = cam.split('.')[0]
    draw_dir = os.path.join(dir, camera_name)
    try:
        os.makedirs(draw_dir)
    except FileExistsError:
        print("檔案已存在。")
    # cv2.imshow("Repro", image)
    file_name = "reprojection_" + camera_name +'.png'
    save_path = os.path.join(draw_dir, file_name)
    print(save_path)
    cv2.imwrite(save_path, image)

def DrawResult(dir, image, camera_matrix, distortion_coeffs, rvecs, tvecs, cam, markerID):
    kSquareSize = 40
    aruco_axes = cv2.drawFrameAxes(image, camera_matrix, distortion_coeffs, rvecs, tvecs, kSquareSize)
    aruco_axes_t = cv2.aruco.drawAxis(image, camera_matrix, distortion_coeffs, rvecs, tvecs, 0.02)
    aruco_cam_dir = os.path.join(dir, cam.split('.')[0])

    try:
        os.makedirs(aruco_cam_dir)
    except FileExistsError:
        print("檔案已存在。")
    
    file_name = "aruco_axes_" + str(markerID) +'.png'
    save_path = os.path.join(aruco_cam_dir, file_name)
    print(save_path)
    #cv2.imshow("Axes", aruco_axes)
    cv2.imwrite(save_path, aruco_axes)
    #cv2.waitKey(0)


def generateAruco(img_path, aruco_type, aruco_id):
    if ARUCO_DICT.get(aruco_type, None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(aruco_type))
        return

    # load aruco dictionary
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

    print("Generating ArUCo Tag [ Type '{}' ] with [ ID '{}' ]".format(aruco_type, aruco_id))
    tag = np.zeros((300, 300, 1), dtype="uint8")
    #print(int(aruco_id))
    

    # store aruco
    aruco_dir = os.path.join(os.getcwd(), img_path, aruco_type)
    try:
        os.makedirs(aruco_dir)
    except FileExistsError:
        print("檔案已存在。")
    
    for aruco_i in aruco_ids:
        
        aruco_name = os.path.join(aruco_dir, "aruco_" + aruco_i + ".png")
        print(aruco_name)
        cv2.aruco.drawMarker(arucoDict, int(aruco_i), 300, tag, 1)
        cv2.imwrite(aruco_name, tag)
        cv2.imshow("ArUCo Tag", tag)
        cv2.waitKey(0)