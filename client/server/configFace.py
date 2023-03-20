import dlib
import os

"""
人脸识别配置
"""
class FACE_FILE(object):

    shape_predictor_path='alg/faceRec/data/data_dlib/shape_predictor_68_face_landmarks.dat'
    recognition_model_path='alg/faceRec/data/data_dlib/dlib_face_recognition_resnet_model_v1.dat'
    csv_base_path='alg/faceRec/data/csv/features.csv'
    faceData_path='alg/faceRec/data/faceData/'
    points_faceData_path='alg/faceRec/data/faceData_points/'
    faceName_path='alg/faceRec/data/faceName.txt'
    imgs_folder_path=os.listdir("alg/faceRec/data/faceData/")
    font_path = "alg/fonts/MSYH.ttc"

FACE_CONFIG={

    "max_collection_image": 50,
    "get_points_faceData_flag": True,
    "import_all_features_flag":True,
    "face_needTo_update":[x for x in range(1, 2)],          #选择更新脸部的编号，从0开始
    "num_of_person_in_lib":len(FACE_FILE.imgs_folder_path),
    "recognition_threshold":0.43,
    "predictor": dlib.shape_predictor(FACE_FILE.shape_predictor_path),
    "recognition_model": dlib.face_recognition_model_v1(FACE_FILE.recognition_model_path),
    "detector":dlib.get_frontal_face_detector(),
}



