"""
负责收集人脸关键点位
"""
import cv2 as cv
import time
import os
from client.server.configFace import FACE_FILE,FACE_CONFIG


class Collection(object):

    """
    提供两种解析方式：
        1. 通过opencv的VideoCapture 进行读取，然后得到图像
        2. 通过图像进行解析，读取，同时得到框出头像之后的图像
    """
    def __init__(self):


        self.start_time = 0
        self.fps = 0
        self.image = None
        self.face_img = None
        self.face_num = 0
        self.last_face_num = 0
        self.face_num_change_flag = False
        self.quit_flag = False
        self.buildNewFolder = False             # 按下"n"新建文件夹标志位
        self.save_flag = False                  # 按下“s”保存人脸数据标志位
        self.face_flag = False
        self.img_num = 0
        self.collect_face_data = True


    def get_fps(self):
        now = time.time()
        time_period = now - self.start_time
        self.fps = 1.0 / time_period
        self.start_time = now
        color = (0,255,0)
        if self.fps < 15:
            color = (0,0,255)
        cv.putText(self.image, str(self.fps.__round__(2)), (20, 50), cv.FONT_HERSHEY_DUPLEX, 1, color)

    def save_face_image(self,build_path):
        buildFile = build_path
        if(not build_path):
            buildFile = FACE_FILE.faceData_path + 'person_{}'.format(FACE_CONFIG.get("num_of_person_in_lib"))
        if(os.path.exists(buildFile)):
            self.buildNewFolder = True
        else:
            os.makedirs(buildFile)
            FACE_CONFIG["num_of_person_in_lib"] = FACE_CONFIG.get("num_of_person_in_lib") + 1
            print("存放人脸数据的文件夹创建成功！！！")
            self.buildNewFolder = True
        if (self.collect_face_data == True  and self.buildNewFolder == True):
            if (self.face_img.size > 0):
                cv.imwrite(
                    FACE_FILE.faceData_path + 'person_{}/{}.png'.format(FACE_CONFIG.get("num_of_person_in_lib") - 1, self.img_num),
                    self.face_img)
                self.img_num += 1

    def key_scan(self, key):
        if self.collect_face_data == True:
            if self.save_flag == True and self.buildNewFolder == True:
                if self.face_img.size > 0:
                    cv.imwrite(
                        FACE_FILE.faceData_path + 'person_{}/{}.png'.format(FACE_CONFIG.get("num_of_person_in_lib") - 1, self.img_num),
                        self.face_img)
                    self.img_num += 1

            if key == ord('s'):
                self.save_flag = not self.save_flag

            if key == ord('n'):
                os.makedirs(FACE_FILE.faceData_path + 'person_{}'.format(FACE_CONFIG.get("num_of_person_in_lib")))
                FACE_CONFIG["num_of_person_in_lib"] = FACE_CONFIG.get("num_of_person_in_lib")+1
                print("新文件夹建立成功!!")
                self.buildNewFolder = True
        if key == ord('q'): self.quit_flag = True

    def face_detecting(self):
        face_location = []
        all_face_location = []
        faces = FACE_CONFIG.get("detector")(self.image, 0)
        self.face_num = len(faces)

        if self.face_num != self.last_face_num:
            self.face_num_change_flag = True
            # print("脸数改变，由{}张变为{}张".format(self.last_face_num, self.face_num))
            self.check_times = 0
            self.last_face_num = self.face_num
        else:
            self.face_num_change_flag = False

        if len(faces) != 0:
            self.face_flag = True

            for i, face in enumerate(faces):
                face_location.append(face)
                w, h = (face.right() - face.left()), (face.bottom() - face.top())
                left, right, top, bottom = face.left() - w//4, face.right() + w//4, face.top() - h//2, face.bottom() + h//4

                all_face_location.append([left, right, top, bottom])

            return face_location, all_face_location
        else:
            self.face_flag = False

        return None


    def collection_cramer(self, camera,show=True):
        """
        :param camera: 摄像头视频/读取视频
        :param show: 是否要展示框选出头像
        :return:
        当处理完毕之后，将保持到好识别出来的头像
        """
        while camera.isOpened() and not self.quit_flag:
            val, self.image = camera.read()
            if val == False: continue
            key = cv.waitKey(1)
            res = self.face_detecting()
            if res is not None:
                _, all_face_location = res
                for i in range(self.face_num):
                    [left, right, top, bottom] = all_face_location[i]
                    self.face_img = self.image[top:bottom, left:right]
                    cv.rectangle(self.image, (left, top), (right, bottom), (0, 0, 255))

                    if self.collect_face_data == True:
                        cv.putText(self.image, "Face", (int((left + right) / 2) - 50, bottom + 20), cv.FONT_HERSHEY_COMPLEX, 1,
                                   (255, 255, 255))
                self.key_scan(key)
            self.get_fps()
            cv.namedWindow('camera', 0)
            if(show):
                cv.imshow('camera', self.image)

            if(self.img_num>=FACE_CONFIG.get("max_collection_image")):
                print("采集完毕！！！")
                break

        camera.release()
        cv.destroyAllWindows()

    def collection_images(self,images,save_path=None):
        """
        :param images: 图片，类型是图片数组，并且对象是opencv读取的图像对象
        :param save_path: 图片保存的路径
        :return:
        如果，传入的图像路径为None的话，那么这里就执行默认的策略，也就是增量修改人物模型
        如果传入的图像有路径的话，那么就直接保存到那里面去
        """

        for image in images:
            self.image = image
            res = self.face_detecting()
            if res is not None:
                _, all_face_location = res
                for i in range(self.face_num):
                    [left, right, top, bottom] = all_face_location[i]
                    self.face_img = self.image[top:bottom, left:right]
                    cv.rectangle(self.image, (left, top), (right, bottom), (0, 0, 255))

                    if self.collect_face_data == True:
                        cv.putText(self.image, "Face", (int((left + right) / 2) - 50, bottom + 20), cv.FONT_HERSHEY_COMPLEX, 1,
                                   (255, 255, 255))
                self.save_face_image(save_path)



