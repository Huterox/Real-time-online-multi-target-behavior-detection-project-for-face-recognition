import numpy as np
import os,cv2,time,torch,natsort,random,pytorchvideo,warnings,threading
from pytorchvideo.data.encoded_video import EncodedVideo

warnings.filterwarnings("ignore",category=UserWarning)
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,)

from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from client.server.alg.poseRec.deep_sort.deep_sort import DeepSort
from client.server.configPose import DECTION_CONFIG
import shutil
from queue import Queue

class PoseRec(object):
    """
    主要提供两种类型的方式
        1. 从视频当中读取信息,并完成识别
        2. 实时读取视频信息,并且实时保存图像
    """
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.conf = DECTION_CONFIG.yolo_conf
        self.model.iou = DECTION_CONFIG.yolo_iou
        self.model.max_det = 200
        # 是否要一直实时开启读取摄像头
        self.going = True
        # 临时存储视频的路径
        # 数据处理完毕
        self.finish_process = False
        self.read_camera_videos = Queue(DECTION_CONFIG.max_queue_size)
        # 存在临时视频
        self.exitTempStreamVideo = False
        # 待处理的数据
        self.read_process_data = Queue(DECTION_CONFIG.max_queue_size)

        self.device = DECTION_CONFIG.yolo_device
        self.imsize = DECTION_CONFIG.yolo_imsize
        # 加载slowfast模型
        self.video_model = slowfast_r50_detection(True).eval().to(self.device)
        # 加载多目标跟踪模型
        self.deepsort_tracker = DeepSort(DECTION_CONFIG.deepsort_ckpt)
        self.ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map(DECTION_CONFIG.deepsort_pb)
        self.coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

    def ava_inference_transform(self,clip, boxes,
        num_frames = 32,
        #if using slowfast_r50_detection, change this to 32, 4 for slow
        crop_size = 640,
        data_mean = DECTION_CONFIG.data_mean,
        data_std = DECTION_CONFIG.data_std,
        slow_fast_alpha = 4,
        #if using slowfast_r50_detection, change this to 4, None for slow
    ):
        boxes = np.array(boxes)
        roi_boxes = boxes.copy()
        clip = uniform_temporal_subsample(clip, num_frames)
        clip = clip.float()
        clip = clip / 255.0
        height, width = clip.shape[2], clip.shape[3]
        boxes = clip_boxes_to_image(boxes, height, width)
        clip, boxes = short_side_scale_with_boxes(clip,size=crop_size,boxes=boxes,)
        clip = normalize(clip,
            np.array(data_mean, dtype=np.float32),
            np.array(data_std, dtype=np.float32),)
        boxes = clip_boxes_to_image(boxes, clip.shape[2],  clip.shape[3])
        if slow_fast_alpha is not None:
            fast_pathway = clip
            slow_pathway = torch.index_select(clip,1,
                torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
            clip = [slow_pathway, fast_pathway]

        return clip, torch.from_numpy(boxes), roi_boxes

    def plot_one_box(self,x, img, color=(100,100,100), text_info="None",
                     velocity=None,thickness=1,fontsize=0.5,fontthickness=1):
        # Plots one bounding box on image img
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
        t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize , fontthickness+2)[0]
        cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1]*1.45)), color, -1)
        cv2.putText(img, text_info, (c1[0], c1[1]+t_size[1]+2),
                    cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255,255,255], fontthickness)
        return img

    def deepsort_update(self,Tracker,pred,xywh,np_img):
        outputs = Tracker.update(xywh, pred[:,4:5],pred[:,5].tolist(),cv2.cvtColor(np_img,cv2.COLOR_BGR2RGB))
        return outputs

    def visualize_yolopreds(self,yolo_preds,id_to_ava_labels,color_map,save_folder,show=False):
        """
        显示yolo预测的结果，绘制图像
        :param yolo_preds:
        :param id_to_ava_labels:
        :param color_map:
        :param save_folder:
        :return:
        """
        for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):
            im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            if pred.shape[0]:
                for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                    if int(cls) != 0:
                        ava_label = ''
                    elif trackid in id_to_ava_labels.keys():
                        ava_label = id_to_ava_labels[trackid].split(' ')[0]
                        """
                        在这里得到动作信息和对应的bounding box
                        """
                        # print(ava_label,box)
                    else:
                        ava_label = 'Unknow'
                    text = '{} {} {}'.format(int(trackid),yolo_preds.names[int(cls)],ava_label)
                    color = color_map[int(cls)]

                    im = self.plot_one_box(box,im,color,text)
            if(show):
                cv2.namedWindow('camera', 0)
                cv2.imshow('camera',im)
                cv2.waitKey(1)
            cv2.imwrite(os.path.join(save_folder,yolo_preds.files[i]),im)

    def extract_video(self,video_path,img_folder):
        """
        读取视频
        :param video_path:
        :param img_folder:
        :return:
        """
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                index = int(cap.get(1))
                cv2.imwrite(os.path.join(img_folder,f'{index}.jpg'),frame)
            else:
                break
        cap.release()

    def clean_folder(self,folder_name):
        if(os.path.exists(folder_name)):
            shutil.rmtree(folder_name)
            os.makedirs(folder_name)
        else:
            os.makedirs(folder_name)

    def readTimeCamera(self,video_path,CameraName):
        """
        读取摄像头,并且按照帧数读取,并且把这读取的图像
        先合成视频,因为这边会用到这个pytorchvideo优化
        :param CameraName:
        :return:
        """

        def readCamera(video_path,CameraName):

            camera = cv2.VideoCapture(video_path)
            wait_time = int((1 / DECTION_CONFIG.realTimeFps) * 1000)
            count = 0
            imgs = []
            split_count = 0
            try:
                while camera.isOpened() and self.going:
                    success,img = camera.read()
                    imgs.append(img)
                    count+=1
                    if(count==DECTION_CONFIG.realTimeFps*DECTION_CONFIG.tempStream):
                        # 此时读取了长度为tempStream秒的视频
                        vide_save_path = DECTION_CONFIG.output + CameraName
                        if (not os.path.exists(vide_save_path)):
                            os.makedirs(vide_save_path)
                        split_count+=1
                        # 这里达到队列长度之后进行重复覆盖
                        if(split_count==DECTION_CONFIG.max_queue_size):
                            split_count = 1
                        vide_save_path = vide_save_path + "/" + CameraName +"-"+str(split_count)+ ".mp4"

                        height = len(img)
                        width = len(img[0])
                        video = cv2.VideoWriter(vide_save_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                                DECTION_CONFIG.realTimeFps, (width, height))
                        for img in imgs:
                            video.write(img)
                        video.release()
                        imgs = []
                        count=0
                        self.exitTempStreamVideo = True
                        # 如果没有消耗掉，这里会进行一个等待，阻塞
                        self.read_camera_videos.put(vide_save_path)
                    key = cv2.waitKey(wait_time)
                    if key == ord('q'):
                        self.going = False
                return
            except Exception as e:
                self.going = False
                print(e)
                return
            finally:
                camera.release()
                print("摄像头终止并释放")
                return

        """
        开启多线程进行调用
        """
        t = threading.Thread(target=readCamera,args=(video_path,CameraName,))
        t.start()
        # readCamera(video_path,CameraName)

    def readTimeProcessing(self,CameraName):
        """
        在这里完成算法处理
        :return:
        """
        def processing(CameraName):

            while(1):

                if(not self.going and self.read_camera_videos.empty()):
                    print("---计算完毕---")
                    self.finish_process = True
                    return

                # 加载识别类型
                if DECTION_CONFIG.yolo_classes:
                    self.model.classes = DECTION_CONFIG.yolo_classes
                # 加载路径视频
                if(self.read_camera_videos.empty() and not self.exitTempStreamVideo):
                    continue
                vide_path = self.read_camera_videos.get()
                # pytorch提供的动作识别器,就是因为这个所以我们还需要存储一下视频文件
                # 虽然会消耗IO的时间,但是这玩意有对视频的优化

                video = pytorchvideo.data.encoded_video.EncodedVideo.from_path(vide_path)
                # 这里还是和先前一样
                img_path = DECTION_CONFIG.streamTempBaseChannel + "/" + CameraName + "/01"
                self.clean_folder(img_path)
                os.makedirs(img_path, exist_ok=True)
                # 对视频进行切分为图片
                self.extract_video(vide_path, img_path)
                imgnames = natsort.natsorted(os.listdir(img_path))

                process_batch_size = DECTION_CONFIG.realTimeFps  # 10 ~ 30 should be fine, the bigger, the faster
                video_clip_length = DECTION_CONFIG.solowfast_video_clip_length  # set 0.8 or 1 or 1.2
                frames_per_second = DECTION_CONFIG.solowfast_frames_per_second  # usually set 25 or 30

                for i in range(0, len(imgnames), process_batch_size):
                    imgs = [os.path.join(img_path, name) for name in imgnames[i:i + process_batch_size]]
                    yolo_preds = self.model(imgs, size=self.imsize)
                    mid = (i + process_batch_size / 2) / frames_per_second
                    video_clips = video.get_clip(mid - video_clip_length / 2, mid + video_clip_length / 2 - 0.04)
                    video_clips = video_clips['video']
                    if video_clips is None:
                        continue
                    # print("*"*100)
                    # print(i / frames_per_second, video_clips.shape, len(imgs))
                    deepsort_outputs = []
                    for i in range(len(yolo_preds.pred)):
                        temp = self.deepsort_update(self.deepsort_tracker, yolo_preds.pred[i].cpu(),
                                                    yolo_preds.xywh[i][:, 0:4].cpu(), yolo_preds.ims[i])
                        if len(temp) == 0:
                            temp = np.ones((0, 8))
                        deepsort_outputs.append(temp.astype(np.float32))
                    yolo_preds.pred = deepsort_outputs
                    id_to_ava_labels = {}
                    if yolo_preds.pred[len(imgs) // 2].shape[0]:
                        inputs, inp_boxes, _ = self.ava_inference_transform(video_clips,
                                                                            yolo_preds.pred[len(imgs) // 2][:, 0:4],
                                                                            crop_size=self.imsize)
                        inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)
                        if isinstance(inputs, list):
                            inputs = [inp.unsqueeze(0).to(self.device) for inp in inputs]
                        else:
                            inputs = inputs.unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            slowfaster_preds = self.video_model(inputs, inp_boxes.to(self.device))
                            slowfaster_preds = slowfaster_preds.cpu()
                        for tid, avalabel in zip(yolo_preds.pred[len(imgs) // 2][:, 5].tolist(),
                                                 np.argmax(slowfaster_preds, axis=1).tolist()):
                            id_to_ava_labels[tid] = self.ava_labelnames[avalabel + 1]

                    """
                    在这里,我们把这个结果放在队列当中
                    """
                    print("-" * 100)
                    print("已完成当前视频计算，剩余：",self.read_camera_videos.qsize())
                    print("-" * 100)
                    self.read_process_data.put((yolo_preds, id_to_ava_labels))
                    self.exitTempStreamVideo = False

        """
        开启多线程进行调用
        """
        t = threading.Thread(target=processing, args=(CameraName,))
        t.start()


    def readTime_visualize_yolopreds(self,CameraName,show=False,process=None):
        """
        显示yolo预测的结果，绘制图像,咱们在这里进行逻辑处理
        在这里如果需要进行逻辑处理的话,在这里传入一个process方法
        这个方法需要接收三个参数,第一个参数是,当前的动作,另一个是当前对象的bounding box 已经image对象
        :param yolo_preds:
        :param id_to_ava_labels:
        :param color_map:
        :param save_folder:
        :return:
        """
        def visulize_readTime(CameraName,show=False,process=None):

            while(1):
                if(self.finish_process and self.read_process_data.empty()):
                    print("---计算处理完毕---")
                    break
                data = self.read_process_data.get()

                print("*"*100)
                print("正在处理实时结果，剩余：",self.read_process_data.qsize())
                print("*" * 100)
                yolo_preds = data[0]
                id_to_ava_labels = data[1]
                for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):
                    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
                    if pred.shape[0]:
                        for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                            if int(cls) != 0:
                                ava_label = ''
                            elif trackid in id_to_ava_labels.keys():
                                ava_label = id_to_ava_labels[trackid].split(' ')[0]
                                """
                                在这里得到动作信息和对应的bounding box
                                并且处理一些逻辑
                                """
                                if(process!=None):
                                    process(im,ava_label,box)
                            else:
                                ava_label = 'Unknow'
                            text = '{} {} {}'.format(int(trackid),yolo_preds.names[int(cls)],ava_label)
                            color = self.coco_color_map[int(cls)]

                            im = self.plot_one_box(box,im,color,text)
                    if(show):
                        cv2.namedWindow('camera', 0)
                        cv2.imshow('camera',im)
                        cv2.waitKey(1)
            folder_name_temp = DECTION_CONFIG.output + CameraName
            if (os.path.exists(folder_name_temp)):
                shutil.rmtree(folder_name_temp)

            print("!"*100)
            print("视频处理完毕！")
            print("!" * 100)


        t = threading.Thread(target=visulize_readTime, args=(CameraName,show,process,))
        t.start()

    def detect_from_video_realTime(self,camera,CameraName,show=False,process=None):
        """
        这里需要传入一个camera对象,从而实现实时读取
        :param camera:
        :param CameraName:
        :param show:
        :param save_path:
        :return:
        """
        # 实时读取摄像头
        self.readTimeCamera(camera,CameraName)
        # 进行处理运算
        self.readTimeProcessing(CameraName)
        # 对预测结果进行处理
        self.readTime_visualize_yolopreds(CameraName,show,process)


    def detect_form_video(self,CameraName,show=False):

        """
        这里的代码执行流程是这样子的:
            1. 读取到视频当中所有的图像
            2. 先通过yolo算法,获取到多个目标
            3. 通过deep_sort对这些目标进行跟踪
            4. 通过slowfast识别出对应的目标动作
            5. 调用visualize_yolopreds处理出识别出来的结果,同时绘制图像
            6. 将处理完毕之后的视频,再次进行合并为一个视频,并输出到指定文件中
        这里有个毛病就是,需要在处理完一个batch之后,你才能看到这个视频,并且每次都有卡顿
        这里实现的方式有延迟.
        :param CameraName:
        :return:
        """
   
        # self.model = torch.hub.load('path/to/yolov5', 'custom', path='path/to/best.pt', source='local')
       
        if DECTION_CONFIG.yolo_classes:
            self.model.classes = DECTION_CONFIG.yolo_classes


        # 读取视频
        video_path = DECTION_CONFIG.input_date
        # pytorch提供的动作识别器

        video = pytorchvideo.data.encoded_video.EncodedVideo.from_path(video_path)

        img_path= DECTION_CONFIG.streamTempBaseChannel+"/"+CameraName+"/01"
        self.clean_folder(img_path)
        os.makedirs(img_path,exist_ok=True)
        print("extracting video...")
        # 对视频进行切分为图片
        self.extract_video(video_path,img_path)
        imgnames=natsort.natsorted(os.listdir(img_path))

        save_path=DECTION_CONFIG.streamTempBaseChannel+"/"+CameraName+"/02"
        self.clean_folder(save_path)
        os.makedirs(save_path,exist_ok=True)

        process_batch_size = DECTION_CONFIG.solowfast_process_batch_size    # 10 ~ 30 should be fine, the bigger, the faster
        video_clip_length = DECTION_CONFIG.solowfast_video_clip_length   # set 0.8 or 1 or 1.2
        frames_per_second = DECTION_CONFIG.solowfast_frames_per_second     # usually set 25 or 30
        print("processing...")
        a=time.time()
        for i in range(0,len(imgnames),process_batch_size):
            imgs=[os.path.join(img_path,name) for name in imgnames[i:i+process_batch_size]]
            yolo_preds=self.model(imgs, size=self.imsize)
            mid=(i+process_batch_size/2)/frames_per_second
            video_clips=video.get_clip(mid - video_clip_length/2, mid + video_clip_length/2 - 0.04)
            video_clips=video_clips['video']
            if video_clips is None:
                continue
            print(i/frames_per_second,video_clips.shape,len(imgs))
            deepsort_outputs=[]
            for i in range(len(yolo_preds.pred)):
                temp=self.deepsort_update(self.deepsort_tracker,yolo_preds.pred[i].cpu(),yolo_preds.xywh[i][:,0:4].cpu(),yolo_preds.ims[i])
                if len(temp)==0:
                    temp=np.ones((0,8))
                deepsort_outputs.append(temp.astype(np.float32))
            yolo_preds.pred=deepsort_outputs
            id_to_ava_labels={}
            if yolo_preds.pred[len(imgs)//2].shape[0]:
                inputs,inp_boxes,_= self.ava_inference_transform(video_clips,yolo_preds.pred[len(imgs)//2][:,0:4],crop_size=self.imsize)
                inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
                if isinstance(inputs, list):
                    inputs = [inp.unsqueeze(0).to(self.device) for inp in inputs]
                else:
                    inputs = inputs.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    slowfaster_preds = self.video_model(inputs, inp_boxes.to(self.device))
                    slowfaster_preds = slowfaster_preds.cpu()
                for tid,avalabel in zip(yolo_preds.pred[len(imgs)//2][:,5].tolist(),np.argmax(slowfaster_preds,axis=1).tolist()):
                    id_to_ava_labels[tid]=self.ava_labelnames[avalabel+1]

            self.visualize_yolopreds(yolo_preds,id_to_ava_labels,self.coco_color_map,save_path,show)

        print("total cost: {:.3f}s, video clips length: {}s".format(time.time()-a,len(imgnames)/frames_per_second))

        vide_save_path = DECTION_CONFIG.output+CameraName
        if(not os.path.exists(vide_save_path)):
            os.makedirs(vide_save_path)
        vide_save_path = vide_save_path+"/"+CameraName+".mp4"
        img_list=natsort.natsorted(os.listdir(save_path))
        im=cv2.imread(os.path.join(save_path,img_list[0]))
        height, width = im.shape[0], im.shape[1]
        video = cv2.VideoWriter(vide_save_path,cv2.VideoWriter_fourcc(*'mp4v'), 25, (width,height))

        for im_name in img_list:
            img = cv2.imread(os.path.join(save_path,im_name))
            video.write(img)
        video.release()

        self.clean_folder(img_path)
        self.clean_folder(save_path)
        print('saved video to:', vide_save_path)


