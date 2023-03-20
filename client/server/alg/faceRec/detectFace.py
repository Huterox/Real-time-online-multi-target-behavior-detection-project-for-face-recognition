"""
负责
"""

import numpy as np
import csv

import cv2 as cv

from client.server.configFace import FACE_CONFIG,FACE_FILE
from client.server.alg.faceRec.collection import Collection
from PIL import Image, ImageDraw, ImageFont

class DetectFace(Collection):

    def __init__(self):

        super(DetectFace, self).__init__()
        self.available_max_face_num = 50
        self.collect_face_data = False
        # 人脸识别过程不采集数据，固定为False
        self.all_features = []
        # 存储库中所有特征向量
        self.check_features_from_cam = []
        # 存储五次检测过程，每次得到的特征向量
        self.person_name = []
        # 存储的人名映射
        self.all_name = []
        # 存储预测到的所有人名
        self.all_face_location = None
        # 存储一帧中所有人脸的坐标
        self.middle_point = None
        # 存储一张人脸的中心点坐标
        self.last_frame_middle_point = []
        # 存储上一帧所有人脸的中心点坐标
        self.all_e_distance = []
        # 存储当前人脸与库中所有人脸特征的欧氏距离
        self.last_now_middlePoint_eDistance = [66666] * (self.available_max_face_num + 10)
        # 存储这帧与上一帧每张人脸中心点的欧氏距离
        self.init_process()
        for i in range(self.available_max_face_num):
            self.all_e_distance.append([])
            self.person_name.append([])
            self.check_features_from_cam.append([])
            self.last_frame_middle_point.append([])

    def get_feature_in_csv(self):
        # 获得库内所有特征向量
        datas = csv.reader(open(FACE_FILE.csv_base_path, 'r'))
        for row in datas:
            for i in range(128):
                row[i] = float(row[i])

            self.all_features.append(row)

    def get_faceName(self):
        # 所有对应的人名
        with open(FACE_FILE.faceName_path, 'r', encoding='utf-8') as f:
            datas = f.readlines()
            for line in datas:
                self.all_name.append(line[:-1])
            print("已经录入的人名有：{}".format(self.all_name))

    def calculate_EuclideanDistance(self, feature1, feature2):  # 计算欧氏距离
        np_feature1 = np.array(feature1)
        np_feature2 = np.array(feature2)

        EuclideanDistance = np.sqrt(np.sum(np.square(np_feature1 - np_feature2)))

        return EuclideanDistance

    def meadian_filter(self, the_list, num_of_data):
        np_list = np.array(the_list)
        feature_max = np.max(np_list, axis=0)
        feature_min = np.min(np_list, axis=0)
        res = (np.sum(np_list, axis=0) - feature_max - feature_min) / (num_of_data - 2)

        res.tolist()
        return res

    def cv2_add_chinese_text(self, img, text, position, textColor=(0, 0, 255), textSize=30):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            FACE_FILE.font_path, textSize, encoding="utf-8")

        draw.text(position, text, textColor, font=fontStyle)

        return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

    def middle_filter(self, the_list):
        np_list = np.array(the_list)
        return np.median(np_list, axis=0)

    def init_process(self):
        self.get_feature_in_csv()
        self.get_faceName()

    def track_link(self):
        # 让后续帧的序号与初始帧的序号对应
        for index in range(self.face_num):
            self.last_now_middlePoint_eDistance[index] = self.calculate_EuclideanDistance(self.middle_point,
                                                                                          self.last_frame_middle_point[
                                                                                              index])
        this_face_index = self.last_now_middlePoint_eDistance.index(min(self.last_now_middlePoint_eDistance))
        self.last_frame_middle_point[this_face_index] = self.middle_point

        return this_face_index


    def detect_from_image(self,image):

        """
        直接识别一张图片当中的人脸，这个开销是最小的，当然这个精确度嘛，没有直接读取视频好一点
        因为那个的话确定了好几帧的情况，这个的话只是单张图像的。返回的是一个图像的人名列表
        但是实际上的话，我们其实送入的图像其实只会有一个人头像，多目标检测，我们也是把一张图像
        对多个目标进行截取，然后进行识别，因为需要确定每个人物的序。
        :param image:
        :param show:
        :return:
        """
        self.image = image
        # self.image = cv.imread('.test_1.jpg')
        res = self.face_detecting()
        names = []
        if res is not None:
            face, self.all_face_location = res
            max_it = self.face_num if self.face_num < len(res) else len(res)
            for i in range(max_it):
                [left, right, top, bottom] = self.all_face_location[i]
                self.middle_point = [(left + right) / 2, (top + bottom) / 2]

                self.face_img = self.image[top:bottom, left:right]

                cv.rectangle(self.image, (left, top), (right, bottom), (0, 0, 255))

                shape = FACE_CONFIG.get("predictor")(self.image, face[i])

                the_features_from_image = list(
                    FACE_CONFIG.get("recognition_model").compute_face_descriptor(self.image, shape))
                e_distance = []
                for features in self.all_features:
                    e_distance.append(self.calculate_EuclideanDistance(the_features_from_image,
                                                     features))
                if(min(e_distance)<FACE_CONFIG.get("recognition_threshold")):
                    max_index = int(np.argmin(e_distance))
                    names.append(self.all_name[max_index])
        return names

    def detect_from_cam(self,camera):
        """
        这里的话，和我们采集是一样的，就是传入这个camera对象就好了
        :return:
        """
        while camera.isOpened() and not self.quit_flag:
            val, self.image = camera.read()
            if val == False: continue
            key = cv.waitKey(1)

            res = self.face_detecting()  # 0.038s

            if res is not None:
                face, self.all_face_location = res
                for i in range(self.face_num):
                    [left, right, top, bottom] = self.all_face_location[i]
                    self.middle_point = [(left + right) / 2, (top + bottom) / 2]

                    self.face_img = self.image[top:bottom, left:right]

                    cv.rectangle(self.image, (left, top), (right, bottom), (0, 0, 255))

                    shape = FACE_CONFIG.get("predictor")(self.image, face[i])  # 0.002s

                    if self.face_num_change_flag == True or self.check_times <= 5:
                        if self.face_num_change_flag == True:  # 人脸数量有变化，重新进行五次检测
                            self.check_times = 0
                            self.last_now_middlePoint_eDistance = [66666 for _ in range(self.available_max_face_num)]
                            for z in range(self.available_max_face_num):
                                self.check_features_from_cam[z] = []

                        if self.check_times < 5:
                            the_features_from_cam = list(
                                FACE_CONFIG.get("recognition_model").compute_face_descriptor(self.image, shape))
                            if self.check_times == 0:  # 初始帧
                                self.check_features_from_cam[i].append(the_features_from_cam)
                                self.last_frame_middle_point[i] = self.middle_point
                            else:
                                this_face_index = self.track_link()  # 后续帧需要与初始帧的人脸序号对应
                                self.check_features_from_cam[this_face_index].append(the_features_from_cam)

                        elif self.check_times == 5:
                            features_after_filter = self.middle_filter(self.check_features_from_cam[i])
                            self.check_features_from_cam[i] = []
                            for person in range(FACE_CONFIG.get("num_of_person_in_lib")):
                                e_distance = self.calculate_EuclideanDistance(self.all_features[person],
                                                                              features_after_filter)

                                self.all_e_distance[i].append(e_distance)

                            if min(self.all_e_distance[i]) < FACE_CONFIG.get("recognition_threshold"):
                                self.person_name[i] = self.all_name[
                                    self.all_e_distance[i].index(min(self.all_e_distance[i]))]
                                # cv.putText(self.image, self.person_name[i],
                                #            (int((left + right) / 2) - 50, bottom + 20),
                                #            cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                                self.image = self.cv2_add_chinese_text(self.image, self.person_name[i],
                                                                       (int((left + right) / 2) - 50, bottom + 10),
                                                                       (0, 0, 255), 25)
                            else:

                                self.person_name[i] = "Unknown"
                    else:
                        this_face_index = self.track_link()
                        self.image = self.cv2_add_chinese_text(self.image, self.person_name[this_face_index],
                                   (int((left + right) / 2) - 50, bottom + 10),
                                   (0, 0, 255), 25)
                self.check_times += 1
                for j in range(self.available_max_face_num):
                    self.all_e_distance[j] = []
                """
                在这里的话，n,s是不会触发的，这里只是用一下这个q而已，也就是退出
                """
                self.key_scan(key)

            self.get_fps()
            cv.namedWindow('camera', 0)
            cv.imshow('camera', self.image)

        camera.release()
        cv.destroyAllWindows()


