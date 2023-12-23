import cv2
import datetime
import numpy as np
import os
import json

# dir = os.path.join(os.getcwd(), "ex_calibration", "221018_test")

def CheckDirExist(file_dir):
    try:
        os.makedirs(file_dir)
    except FileExistsError:
        print("資料夾已存在。")
    
read_dir = os.path.join(os.getcwd(), "matrix")

def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    T = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(T)

    return M_inv

def getDatetime():
    loc_dt = datetime.datetime.today() 
    loc_dt_format = loc_dt.strftime("%Y/%m/%d %H:%M:%S")
    return loc_dt_format

def read_aruco_json(path):
    f = open(path)
    data = json.load(f)

    aruco_42_corners = np.array(data["aruco_42_corners"])
    print(aruco_42_corners)

    dict_points = {}
    for aruco_id, offset in data["dict_points"].items():
        dict_points[aruco_id] = aruco_42_corners + np.array(offset)
    
    print(type(dict_points))
    return dict_points

def writeYAML(cam, rvec, tvec, extrinsic, intrinsic, file_dir):
    CheckDirExist(file_dir)
    filepath = os.path.join(file_dir, cam.split('.')[0] + '.yml')
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)

    fs.write("Extrinsic", extrinsic)
    fs.write("Intrinsic", intrinsic)
    fs.write("rvec", rvec)
    fs.write("tvec", tvec)
    fs.write("date", getDatetime())
    fs.release()

def writeJson(cam, rvec, tvec, extrinsic, intrinsic, file_dir):
    CheckDirExist(file_dir)
    filepath = os.path.join(file_dir, cam.split('.')[0] + '.json')
    data = {}
    data['Extrinsic'] = extrinsic.flatten().tolist()
    data['Intrinsic'] = intrinsic.flatten().tolist()
    data['Rvec'] = rvec.flatten().tolist()
    data['Tvec'] = tvec.flatten().tolist()

    ex_inv = inverse_homogeneoux_matrix(extrinsic)
    data['Ex_inverse'] = ex_inv.flatten().tolist()
    data['data'] = getDatetime()

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    # print(data)

def writeXml(cam, distortion_coeffs,  extrinsic, intrinsic, file_dir):
    CheckDirExist(file_dir)
    filepath = os.path.join(file_dir, cam.split('.')[0] + '.xml')
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
    # distortion_coeffs = np.zeros((14,1))
    ci = np.eye(3,4, dtype=np.float32)
    # print(ci)
    # ex_inv = inverse_homogeneoux_matrix(extrinsic)
    fs.write("CameraMatrix", extrinsic[:3,:])
    # fs.write("CameraMatrix", ex_inv[:3,:])
    fs.write("Intrinsics", intrinsic)
    fs.write("Distortion", distortion_coeffs)
    fs.write("CameraMatrixInitial", ci)
    # fs.write("date", getDatetime())


    fs.release()

def writeDat(cam, extrinsic, intrinsic, file_dir):
    # rot_trans_c2.dat
    CheckDirExist(file_dir)
    extrinsic_filepath = os.path.join(file_dir, 'rot_trans_' + cam.split('.')[0] + '.dat')
    intrinsic_filepath = os.path.join(file_dir, cam.split('.')[0] + '.dat')
    
    ex_fout = open(extrinsic_filepath, 'w')
    in_fout = open(intrinsic_filepath, 'w')
    
    ''' Extrinsic Matrix '''
    # write rotation matrix
    ex_fout.write('R:')
    ex_fout.write('\n')
    for i, parameters in enumerate(extrinsic):
        # print(parameters)
        if i >= 3:
            break
        ex_fout.write(str(parameters[0]) + ' ' + str(parameters[1]) + ' ' + str(parameters[2]))
        ex_fout.write('\n')
    
    # write translation matrix
    ex_fout.write('T:')
    ex_fout.write('\n')
    for i, parameters in enumerate(extrinsic):
        # print(parameters)
        if i >= 3:
            break
        ex_fout.write(str(parameters[3]))
        ex_fout.write('\n')
    ex_fout.close()
    
    ''' Intrinsic Matrix '''
    in_fout.write('instrinsic:')
    in_fout.write('\n')
    for i, parameters in enumerate(intrinsic):
        in_fout.write(str(parameters[0]) + ' ' + str(parameters[1]) + ' ' + str(parameters[2]))
        in_fout.write('\n')
    
    in_fout.write('distortion:')
    in_fout.write('\n')
    in_fout.write('0 0 0 0 0')
    in_fout.write('\n')
    
    in_fout.close()


    