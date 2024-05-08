from PIL import Image, ImageDraw, ImageFont
import numpy as np
import numpy.ctypeslib as npct
from ctypes import *
import cv2
import re
from time import sleep
import paramiko
import socket


class Events:
    def __init__(self, event_class):
        """event_class:
        0:默认事件检测流程
        1:(LS)室外大规模采样事件检测流程
        """
        self.event_class = event_class
        self.detection = []
        # 事件检测变量
        self.event_num = 0  # 默认事件号
        self.event_ls_num = 0  # LS事件号
        self.event_count = 0  # 事件所需标签计数器
        self.sampling_count = 0  # 当前合格采样次数
        self.samplenum = 12  # 采样合格次数
        self.sampling_flag = False  # 采样判别
        self.sampling2_flag = False  # 采样判别
        self.throat_flag = False  # 采样区域判别
        self.throat2_flag = False  # 采样区域判别
        self.swab_flag = False  # 是否棉签头
        self.intube_flag = False  # 棉签是否在试管中
        self.stick_flag = False  # 是否棉签棒

    def get_result(self, detections):
        self.detection = detections

    def process_event(self):
        if self.event_class == 0:
            if self.event_num == 0:
                self.event_start()
            elif self.event_num == 1:
                self.event_sample()
            elif self.event_num == 2:
                self.event_trackswab()
            elif self.event_num == 3:
                self.event_collectswab()
            elif self.event_num == 4:
                self.event_end()
        elif self.event_class == 1:
            if self.event_ls_num == 0:
                self.event_ls_start()
            elif self.event_ls_num == 1:
                self.event_ls_sample()
            elif self.event_ls_num == 2:
                self.event_ls_end()

    # event_num = 0
    def event_start(self):
        th_flag = False  # 当前帧是否检测到采样区域
        th2_flag = False  # 当前帧是否检测到采样区域2
        sw_flag = False  # 当前帧是否检测到棉签
        for det in self.detection:
            if det['class'] == "throat" and det['conf'] >= 0.7:
                th_flag = True
            elif det['class'] == "throat2" and det['conf'] >= 0.7:
                th2_flag = True
            if det['class'] == "swab" and det['conf'] >= 0.75:
                sw_flag = True
        if th_flag:
            self.throat_flag = True  # 当前帧存在采样区域 throat， 置true
        elif th2_flag:
            self.throat_flag = False
        if (th_flag or th2_flag) and sw_flag:
            self.event_count += 1
            if self.event_count >= 2:  # 采样区域和棉签同时出现1次，触发采样事件(event_num=1)
                self.event_count = 0
                self.event_num = 1

    # event_num = 1
    def event_sample(self):
        s_flag = False  # 当前帧是否检测到采样
        s2_flag = False  # 当前帧是否检测到采样2
        for det in self.detection:
            if det['class'] == "sampling" and det['conf'] >= 0.7:
                s_flag = True
            elif det['class'] == "sampling2" and det['conf'] >= 0.7:
                s2_flag = True
        if s_flag:
            self.sampling_count += 1
            print("采样次数：", +int(self.sampling_count))
        elif s2_flag:
            self.sampling_count += 0.5
            print("采样次数：", + int(self.sampling_count))
        if self.sampling_count >= self.samplenum:  # 采样到 MAX_NUM 次采样成功的帧，判定为采样合格
            self.event_num = 2  # 采样达到要求，触发棉签追踪事件
            self.sampling_count = 0

    # event_num = 2
    def event_trackswab(self):
        sw_flag = False  # 当前帧是否检测到棉签
        in_flag = False  # 当前帧是否检测到棉签放入试管
        for det in self.detection:
            if det['class'] == "swab" and det['conf'] >= 0.6:
                self.event_count = 0  # 检测到棉签则重新计算 intube 次数
                self.swab_flag = True
                sw_flag = True
            if det['class'] == "Intube" and det['conf'] >= 0.65:
                in_flag = True
        if not sw_flag:  # 当前帧没有swab
            self.swab_flag = False
        if (sw_flag is False) and in_flag:  # 该帧是否只存在Intube
            self.event_count += 1
            if self.event_count >= 2:  # 连续只存在Intube出现3次以上，触发棉签收集事件(event_num=3)
                self.event_count = 0
                self.event_num = 3

    # event_num = 3
    def event_collectswab(self):
        st_flag = False  # 当前帧是否检测到棉签棒
        in_flag = False  # 当前帧是否检测到棉签放入试管
        for det in self.detection:
            if det['class'] == "Intube" and det['conf'] >= 0.75:
                self.event_count = 0  # 检测到Intube则重新计算 stick 次数
                self.intube_flag = True
                in_flag = True
            if det['class'] == "stick" and det['conf'] >= 0.7:
                st_flag = True
        if not in_flag:  # 当前帧没有 Intube
            self.intube_flag = False
        if (in_flag is False) and st_flag:  # 该帧是否只存在stick
            self.event_count += 1
            if self.event_count >= 2:  # 连续只存在stick出现3次以上，触发结束事件(event_num=4)
                self.event_count = 0
                self.event_num = 4

    # event_num = 4
    def event_end(self):
        st_flag = False  # 当前帧是否检测到棉签棒
        for det in self.detection:
            if det['class'] == "stick" and det['conf'] >= 0.7:
                self.stick_flag = True
                st_flag = True
            if det['class'] == "swab" and det['conf'] >= 0.8:
                self.event_count = 0  # 检测到棉签头则重新计算 stick 次数
        if not st_flag:  # 当前帧没有 stick
            self.stick_flag = False
        else:
            self.event_count += 1
            if self.event_count >= 3:  # 存在stick出现3次以上，重置事件检测流程
                self.event_count = 0
                self.event_num = 0

    # event_ls_num = 0
    def event_ls_start(self):
        th_flag = False  # 当前帧是否检测到采样区域
        th2_flag = False  # 当前帧是否检测到采样区域2
        sw_flag = False  # 当前帧是否检测到棉签
        for det in self.detection:
            if det['class'] == "throat" and det['conf'] >= 0.7:
                th_flag = True
            elif det['class'] == "throat2" and det['conf'] >= 0.7:
                th2_flag = True
            if det['class'] == "swab" and det['conf'] >= 0.75:
                sw_flag = True
        if th_flag:
            self.throat_flag = True  # 当前帧存在采样区域 throat， 置true
        elif th2_flag:
            self.throat2_flag = True
        else:
            self.throat_flag = False
            self.throat2_flag = False
        if (th_flag or th2_flag) and sw_flag:
            self.event_count += 1
            if self.event_count >= 2:  # 采样区域和棉签同时出现1次，触发采样事件(event_odls_num=1)
                self.event_count = 0
                self.event_ls_num = 1

    # event_ls_num = 1
    def event_ls_sample(self):
        s_flag = False  # 当前帧是否检测到采样
        s2_flag = False  # 当前帧是否检测到采样2
        for det in self.detection:
            if det['class'] == "sampling" and det['conf'] >= 0.7:
                s_flag = True
            elif det['class'] == "sampling2" and det['conf'] >= 0.7:
                s2_flag = True
        if s_flag:
            self.sampling_flag = True
            # self.sampling_count += 1
            # print("采样次数：", +int(self.sampling_count))
        elif s2_flag:
            self.sampling2_flag = True
            # self.sampling_count += 0.5
            # print("采样次数：", + int(self.sampling_count))
        if self.sampling_count >= self.samplenum:  # 采样到 MAX_NUM 次采样成功的帧，判定为采样合格
            self.event_ls_num = 2  # 采样达到要求，触发棉签追踪事件
            self.sampling_count = 0

    # event_num = 2
    def event_ls_end(self):
        st_flag = False  # 当前帧是否检测到棉签棒
        for det in self.detection:
            if det['class'] == "stick" and det['conf'] >= 0.7:
                self.stick_flag = True
                st_flag = True
            if det['class'] == "swab" and det['conf'] >= 0.8:
                self.event_count = 0  # 检测到棉签头则重新计算 stick 次数
        if not st_flag:  # 当前帧没有 stick
            self.stick_flag = False
        else:
            self.event_count += 1
            if self.event_count >= 3:  # 存在stick出现3次以上，重置事件检测流程
                self.event_count = 0
                self.event_ls_num = 0


class Amplifier:
    def __init__(self, detections=[], origin_img=np.zeros((640, 480, 3), dtype='uint8')):
        self.detections = detections
        self.origin_img = origin_img
        self.priority_dic = {'throat': 4, 'throat2': 4, 'sampling': 5, 'sampling2': 5, 'sampling3': 5, 'swab': 2,
                             'Intube': 3, 'stick': 1}
        self.pre_img = origin_img

    def select_object(self):
        res = {}
        # 当前帧没有检测物体，返回空
        if not self.detections:
            return res
        # 初始化
        res = self.detections[0]
        for i, obj in enumerate(self.detections):
            if i == 0:
                continue
            if self.priority_dic[obj['class']] > self.priority_dic[res['class']]:
                res = obj
            elif self.priority_dic[obj['class']] == self.priority_dic[res['class']]:
                if obj['conf'] > res['conf']:
                    res = obj
        return res

    # 返回放大的局部图片
    def get_img(self):
        obj = self.select_object()
        if not obj:
            # return np.zeros(self.origin_img.shape, dtype='uint8'), False
            return self.pre_img, False
        x, y = obj['center_xy']
        # huan_amplification
        amplify_num = 3
        img_h, img_w = self.origin_img.shape[:2]
        h, w = img_h // amplify_num, img_w // amplify_num
        try:
            res_img = self.origin_img[y - h // 2:y + h // 2, x - w // 2:x + w // 2, :]
            res_img = cv2.resize(res_img, dsize=(self.origin_img.shape[1], self.origin_img.shape[0]))
            self.set_preImg(res_img)
            self.set_preImg(res_img)
            return res_img, True
        except:
            # print('error')
            # return np.zeros(self.origin_img.shape, dtype='uint8'), False
            return self.pre_img, False

    def set_detection(self, detection):
        self.detections = detection

    def set_origin_img(self, img):
        self.origin_img = img

    def set_preImg(self, img):
        self.pre_img = img


class AmplifierPhoneCam(Amplifier):
    def __int__(self, detection, origin_img):
        super(AmplifierPhoneCam, self).__int__(detection, origin_img)
        self.img_size = self.origin_img.shape

    def set_dection(self, detection):
        self.detections = detection

    def set_origin_img(self, img):
        self.origin_img = img


class Example:
    def __init__(self):
        img_path_list = ['./example_images/throat.jpg',
                         './example_images/intube.mp4']
        self.gif1 = get_frame(img_path_list[0])
        self.gif2 = get_frame(img_path_list[1])
        self.len_gif1 = len(self.gif1)
        self.len_gif2 = len(self.gif2)
        resize_frame_list(self.gif1, 480, 640)
        resize_frame_list(self.gif2, 480, 640)
        self.gif1_index = 0
        self.gif2_index = 0


def resize_frame_list(gif=[], W=0, H=0):
    for i, _ in enumerate(gif):
        gif[i] = cv2.resize(gif[i], dsize=(int(W), int(H)))


def get_frame(path):
    frame_list = []
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if ret:
            frame_list.append(frame)
        else:
            cap.release()
            break
    return frame_list


def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


class Linux(object):
    # 通过IP, 用户名，密码，超时时间初始化一个远程Linux主机
    def __init__(self, ip, username, password, timeout=30):
        self.ip = ip
        self.username = username
        self.password = password
        self.timeout = timeout
        # transport和chanel
        self.t = ''
        self.chan = ''
        # 链接失败的重试次数
        self.try_times = 3

    # 调用该方法连接远程主机
    def connect(self):
        while True:
            # 连接过程中可能会抛出异常，比如网络不通、链接超时
            try:
                self.t = paramiko.Transport(sock=(self.ip, 22))
                self.t.connect(username=self.username, password=self.password)
                self.chan = self.t.open_session()
                self.chan.settimeout(self.timeout)
                self.chan.get_pty()
                self.chan.invoke_shell()
                # 如果没有抛出异常说明连接成功，直接返回
                print(u'连接%s成功' % self.ip)
                # 接收到的网络数据解码为str
                print(self.chan.recv(65535).decode('utf-8'))
                return
            # 这里不对可能的异常如socket.error, socket.timeout细化，直接一网打尽
            except Exception:
                if self.try_times != 0:
                    print(u'连接%s失败，进行重试' % self.ip)
                    self.try_times -= 1
                else:
                    print(u'重试3次失败，结束程序')
                    exit(1)

    # 断开连接
    def close(self):
        self.chan.close()

        self.t.close()

    # 发送要执行的命令
    def send(self, cmd, pattern):
        cmd += '\r'
        # 通过命令执行提示符来判断命令是否执行完成
        patt = pattern
        p = re.compile(patt)
        result = ''
        # 发送要执行的命令
        self.chan.send(cmd)
        # 回显很长的命令可能执行较久，通过循环分批次取回回显
        while True:
            sleep(0.5)
            ret = self.chan.recv(65535)
            ret = ret.decode('utf-8')
            result += ret
            if p.search(ret):
                print(result)
                return result


class ConnectImageSource(object):
    # 通过IP和port连接手机获取图像
    def __init__(self, source_type, source):
        self.source_type = source_type  # 图像源类型 webcam or phone
        self.source = source  # 0,1.. or url for webcam; 30005 for phone port
        self.ip = self.get_host_ip()  # 主机作为AP热点的ip地址
        self.access_flag = False
        self.img = None
        self.imgRGB = None
        self.send_flag1 = False
        self.send_flag = False

    def access(self, p_input):
        if self.source_type == 'webcam':
            camera = cv2.VideoCapture(self.source)
            while True:  # 重复尝试接入相机
                if camera.isOpened():
                    print("相机接入成功！！!")
                    self.access_flag = True
                    break
                else:
                    camera = cv2.VideoCapture(self.source)
                    print("错误,无法接入，正在重新尝试接入相机！！！")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            while True:
                ret, fram = camera.read()
                if ret:
                    mirror_img = cv2.flip(fram, 1)  # 镜像
                    # 图像要求是RGB，所以此处需要转换图像的格式
                    RGB = cv2.cvtColor(mirror_img, cv2.COLOR_BGR2RGB)
                    RGB = cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB)
                    RGB = cv2.resize(RGB, dsize=(480, 640))
                    if self.send_flag1:
                        p_input.send(RGB)
                        self.send_flag1 = False
                    elif self.send_flag:
                        p_input.send(RGB)
                        self.send_flag = False
        elif self.source_type == 'phone':
            print("Starting socket: TCP...")
            socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建socket

            print("TCP server listen @ %s:%d!" % (self.ip, self.source))
            host_addr = (self.ip, self.source)
            socket_tcp.bind(host_addr)  # 绑定主机的ip地址和端口号
            socket_tcp.listen(1)  # listen函数的参数是监听客户端的个数，这里只监听一个，即只允许与一个客户端创建连接

            print('waiting for connection...')
            socket_con, (client_ip, client_port) = socket_tcp.accept()  # 接收客户端的请求
            print("Connection accepted from %s." % client_ip)
            self.access_flag = True

            while True:
                data1 = self.recv_all(socket_con, 4)  # 接收开始标志
                start = int.from_bytes(data1, byteorder='little', signed=False)
                if start == 303030303:  # 收到开始标志，开始接收图像
                    data2 = self.recv_all(socket_con, 4)  # 接收要接收的数据长度
                    image_length = int.from_bytes(data2, byteorder='little', signed=False)
                    image_bit = self.recv_all(socket_con, image_length)
                    buff = np.frombuffer(image_bit, np.uint8)
                    img_decode = cv2.imdecode(buff, cv2.IMREAD_COLOR)
                    rotated = self.rotate_img(img_decode, 90)  # 竖屏显示
                    mirror_img = cv2.flip(rotated, 1)  # 镜像
                    # 图像要求是RGB，所以此处需要转换图像的格式
                    RGB = cv2.cvtColor(mirror_img, cv2.COLOR_BGR2RGB)
                    RGB = cv2.resize(RGB, dsize=(480, 640))
                    # cv2.imshow("Image", RGB)
                    # # Press "q" to quit
                    # if cv2.waitKey(25) & 0xFF == ord("q"):
                    #     cv2.destroyAllWindows()
                    #     break
                    if self.send_flag1:
                        p_input.send(RGB)
                        self.send_flag1 = False
                    elif self.send_flag:
                        p_input.send(RGB)
                        self.send_flag = False
            socket_tcp.close()

    # 接受图片大小的信息
    def recv_all(self, m_sock, m_count):
        buf = bytes()
        while m_count:
            newbuf = m_sock.recv(m_count)
            if not newbuf: return None
            buf += newbuf
            m_count -= len(newbuf)
        return buf

    def get_host_ip(self):
        """
        查询本机ip地址
        :return: ip
        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        return ip

    def rotate_img(self, image, angle):
        (h, w) = image.shape[:2]  # grab the dimensions of the image and then determine the
        (cX, cY) = (w // 2, h // 2)  # center
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))


class Detector:
    def __init__(self, model_path, dll_path):
        self.yolov5 = CDLL(dll_path)
        self.yolov5.Detect.argtypes = [c_void_p, c_int, c_int, POINTER(c_ubyte),
                                       npct.ndpointer(dtype=np.float32, ndim=2, shape=(50, 6), flags="C_CONTIGUOUS")]
        self.yolov5.Init.restype = c_void_p
        self.yolov5.Init.argtypes = [c_void_p]
        self.yolov5.cuda_free.argtypes = [c_void_p]
        self.c_point = self.yolov5.Init(model_path)
        self.name = ['throat', 'sampling', 'swab', 'Intube', 'stick', 'throat2', 'sampling2']

    def predict(self, img):
        rows, cols = img.shape[0], img.shape[1]
        res_arr = np.zeros((50, 6), dtype=np.float32)
        self.yolov5.Detect(self.c_point, c_int(rows), c_int(cols), img.ctypes.data_as(POINTER(c_ubyte)), res_arr)
        self.bbox_array = res_arr[~(res_arr == 0).all(1)]
        return self.bbox_array

    def visualize(self, img, bbox_array, conf_thres):
        # 用于存放结果
        detections = []
        for temp in bbox_array:
            if float(temp[5]) >= conf_thres:
                bbox = [int(temp[0]), int(temp[1]), int(temp[2]), int(temp[3])]  # xywh
                clas = int(temp[4])
                cls = self.name[clas]
                conf = float(temp[5])
                cxcy = [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2]
                detections.append({'class': cls, 'conf': conf, 'center_xy': cxcy, 'position': bbox})
                cv2.rectangle(img, (int(temp[0]), int(temp[1])), (int(temp[0] + temp[2]), int(temp[1] + temp[3])),
                              (105, 237, 249), 2)
                img = cv2.putText(img, str(cls) + " " + str(round(conf, 2)), (int(temp[0]), int(temp[1]) - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 237, 249), 1)
        return img, detections

    def free(self):
        self.yolov5.cuda_free(self.c_point)


class GlobalVar:
    def __init__(self, person_num):
        self.person_num = person_num
        self.current_person_num = 0
        self.sample_state = False

    def re_initstate(self):
        self.sample_state = False