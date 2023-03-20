"""
负责读取采集到的人脸图像，然后去构建人脸对应的信息
"""
import cv2 as cv
import os
import numpy as np
import csv

from tqdm import tqdm
import shutil
from client.server.configFace import FACE_FILE,FACE_CONFIG

class BuildFace():

    def write2csv(self,data, mode):
        """
        更新csv文件当中的数据（这里面存储的是我们人脸的特征）
        :param data:
        :param mode:
        :return:
        """
        with open(FACE_FILE.csv_base_path, mode, newline='') as wf:
            csv_writer = csv.writer(wf)
            csv_writer.writerow(data)

    def get_features_from_csv(self):

        features_in_csv = []
        with open(FACE_FILE.csv_base_path, 'r') as rf:
            csv_reader = csv.reader(rf)
            for row in csv_reader:
                for i in range(0, 128):
                    row[i] = float(row[i])

                features_in_csv.append(row)
            return features_in_csv

    def save_select_in_csv(self,data):
        """
        选择性更新人脸数据
        :param data:
        :return:
        """
        features_in_csv = self.get_features_from_csv()
        with open(FACE_FILE.csv_base_path, 'w', newline='') as wf:
            csv_writer = csv.writer(wf)
            for index, i in enumerate(FACE_CONFIG.get("face_needTo_update")):
                features_in_csv[i] = data[index]
            csv_writer.writerow(features_in_csv[0])

        with open(FACE_FILE.csv_base_path, 'a+', newline='') as af:
            csv_writer = csv.writer(af)
            for j in range(1, len(features_in_csv)):
                csv_writer.writerow(features_in_csv[j])

        print("csv文件更新完成!!")

    def get_128_features(self,person_index):
        """
        :param person_index:  person_index代表第几个人脸数据文件夹
        :return:
        """
        num = 0
        features = []
        imgs_folder = FACE_FILE.imgs_folder_path[person_index]
        points_faceImage_path = FACE_FILE.points_faceData_path + imgs_folder

        imgs_path = FACE_FILE.faceData_path + imgs_folder + '/'
        list_imgs = os.listdir(imgs_path)
        imgs_num = len(list_imgs)

        if os.path.exists(FACE_FILE.points_faceData_path + imgs_folder):
            shutil.rmtree(points_faceImage_path)
        os.makedirs(points_faceImage_path)
        print("人脸点图文件夹建立成功!!")
        with tqdm(total=imgs_num) as pbar:
            pbar.set_description(str(imgs_folder))
            for j in range(imgs_num):
                image = cv.imread(os.path.join(imgs_path, list_imgs[j]))

                faces = FACE_CONFIG.get("detector")(image, 1)
                if len(faces) != 0:
                    for z, face in enumerate(faces):
                        shape = FACE_CONFIG.get("predictor")(image, face)
                        w, h = (face.right() - face.left()), (face.bottom() - face.top())
                        left, right, top, bottom = face.left() - w // 4, face.right() + w // 4, face.top() - h // 2, face.bottom() + h // 4
                        im = image
                        cv.rectangle(im, (left, top), (right, bottom), (0, 0, 255))
                        cv.imwrite(points_faceImage_path + '/{}.png'.format(j), im)

                        if (FACE_CONFIG.get("get_points_faceData_flag") == True):
                            for p in range(0, 68):
                                cv.circle(image, (shape.part(p).x, shape.part(p).y), 2, (0,0,255))
                            cv.imwrite(points_faceImage_path + '/{}.png'.format(j), image)
                        the_features = list(FACE_CONFIG.get("recognition_model").compute_face_descriptor(image, shape)) # 获取128维特征向量
                        features.append(the_features)
                        num += 1
                pbar.update(1)
        np_f = np.array(features)
        res = np.median(np_f, axis=0)
        return res

    def building_form_config(self):

        if (FACE_CONFIG.get("import_all_features_flag") == True):
            self.building_all()
        else:
            peoples = FACE_CONFIG.get("face_needTo_update")
            self.building_select(peoples)

    def building_all(self):
        res = self.get_128_features(person_index=0)
        self.write2csv(res, 'w')
        for i in range(1, FACE_CONFIG.get("num_of_person_in_lib")):
            res = self.get_128_features(person_index=i)
            self.write2csv(res, 'a+')

    def building_select(self,peoples):
        """
        更新某几个人脸，传入对应的下标编号,例如：[0,2,4]
        :param peoples:
        :return:
        """
        select_res = []
        for i in peoples:
            res = self.get_128_features(person_index=i)
            select_res.append(res)
        self.save_select_in_csv(select_res)

