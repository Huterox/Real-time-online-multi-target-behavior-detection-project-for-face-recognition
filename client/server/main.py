
import cv2 as cv

from client.server.alg.faceRec.buildFace import BuildFace
from client.server.alg.faceRec.collection import Collection
from client.server.alg.faceRec.detectFace import DetectFace
from client.server.alg.poseRec.poseRec import PoseRec


def collection_face():

    cam = cv.VideoCapture(0)
    Collection().collection_cramer(cam)

    cam.release()
    cv.destroyAllWindows()
    print("采集完毕,程序退出!!")

def build_face():
    build = BuildFace()
    build.building_all()

def detect_face():

    cam = cv.VideoCapture(0)
    # im = cv.imread(r'C:\Users\31395\Desktop\peoplePose\client\server\alg\faceRec\data\faceData\person_0\10.png')
    process = DetectFace()
    # print(process.detect_from_image(im))
    process.detect_from_cam(cam)


def detect_pose():

    pose = PoseRec()
    # source_path = r"C:\Users\31395\Desktop\peoplePose\temp\yolo_slowfast\video\test_person.mp4"
    source_path = 0
    # pose.detect_form_video("Test01",True)
    pose.detect_from_video_realTime(source_path,"Test02",True,process=None)


if __name__ == '__main__':
    # collection_face()
    # build_face()
    # detect_face()
    detect_pose()