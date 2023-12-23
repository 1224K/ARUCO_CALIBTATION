import cv2
import numpy as np
from scipy.linalg import svd
#from openpose import detector

def solveSVD(keypoint_2d, camera_matrix):
    #
    
    print(camera_matrix)
    for i in camera_matrix:
        A.append(keypoint_2d[0] * camera_matrix[i][2])
        A.append(keypoint_2d[1] * camera_matrix[i][0])

    A = np.array()

    U, S, V = svd(camera_matrix)
    print(U)
    print(S)
    z = np.zeros(255, 255, 1)

    c = np.dot(U.T, z)
    
    w = np.dot(np.diag(1/S),c)

    keypoint_3d = np.dot(V.conj().T, w)
    keypoint_3d = np.linalg.solve(camera_matrix,z)
    return keypoint_3d
