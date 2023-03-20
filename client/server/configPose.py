"""
目标检测配置
"""
import os


class DECTION_CONFIG():
    #test imgs folder or video or camera
    input_date = r"C:\Users\31395\Desktop\peoplePose\temp\yolo_slowfast\video\test_person.mp4"
    #folder to save result imgs, can not use input folder,视频保存路径
    output = "output/video/"
    #inference size (pixels)
    yolo_imsize = 640
    #object confidence threshold
    yolo_conf = 0.4
    #IOU threshold for NMS
    yolo_iou = 0.4
    #cuda device, i.e. 0 or 0,1,2,3 or cpu
    yolo_device = "cuda"
    #默认已经设置好了是cooc数据集
    yolo_classes = None

    yolo_weight = ""

    #10 ~ 30 should be fine, the bigger, the faster
    solowfast_process_batch_size = 25
    # set 0.8 or 1 or 1.2
    solowfast_video_clip_length = 1.2
    #usually set 25 or 30
    solowfast_frames_per_second = 25

    data_mean = [0.45, 0.45, 0.45]
    data_std = [0.225, 0.225, 0.225]

    deepsort_ckpt = "alg/poseRec/deep_sort/deep_sort/deep/checkpoint/ckpt.t7"
    deepsort_pb = "alg/poseRec/selfutils/temp.pbtxt"

    streamTempBaseChannel = "/alg/poseRec/data/tempChannel"

    # 设置实时处理的FPS
    realTimeFps = 10
    # 最大队列长度
    max_queue_size = 512
    # 每2秒左右存储一次视频,用于实时视频检测
    tempStream = 2