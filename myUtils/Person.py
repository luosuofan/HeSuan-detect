from myUtils.myUtils import resize_frame_list, get_frame
import numpy as np


class Person():
    def __init__(self):
        self.sampling_count = 0
        self.flag_1 = False   # 是否棉签头
        self.flag_2 = False   # 棉签是否在试管中
        self.flag_3 = False   # 是否棉签棒
        self.qrstate = False  # 核酸码状态
        self.uistate = False  # 显示界面状态
        self.send1toyolo = False  # 发送1帧给yolo
        self.sendtoyolo = False  # 发送all帧给yolo

        img_path_list = ['./example_images/throat.jpg',
                         './example_images/intube.mp4']
        self.gif1 = get_frame(img_path_list[0])
        self.gif2 = get_frame(img_path_list[1])
        self.len_gif1 = len(self.gif1)
        self.len_gif2 = len(self.gif2)
        resize_frame_list(self.gif1, 640, 480)
        resize_frame_list(self.gif2, 640, 480)
        self.gif1_index = 0
        self.gif2_index = 0
        self.pre_amplify_img = np.zeros((480, 640, 3), dtype='uint8')

    def re_init(self):
        self.sampling_count = 0
        self.flag_1 = False
        self.flag_2 = False
        self.flag_3 = False
        self.qrstate = False
        self.uistate = False
        self.send1toyolo = False
        self.sendtoyolo = False
