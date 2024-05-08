import json
import socket
import threading
import cv2
import time
import numpy as np
import numpy.ctypeslib as npct
from ctypes import *
from multiprocessing import Pipe


class ConnectImageSource:
    # 通过IP和port连接手机获取图像
    def __init__(self, source_type, source):
        self.source_type = source_type  # 图像源类型 webcam or phone
        self.source = source  # 30005 for phone port
        self.ip = self.get_host_ip()  # 主机作为AP热点的ip地址
        # self.ip = '127.0.0.1'
        self.access_flag = False
        self.img = None
        self.imgRGB = None
        self.send_flag1 = False
        self.send_flag = False

    def access(self):
        socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建socket
        host_addr = (self.ip, self.source)
        socket_tcp.bind(host_addr)  # 绑定主机的ip地址和端口号
        socket_tcp.listen(3)  # listen函数的参数是监听客户端的个数

        if self.source_type == 'webcam':
            print("Waiting webcam connect...")
            socket_pc_img, (pc_img_ip, pc_img_port) = socket_tcp.accept()  # 接收客户端的请求
            print("Destination1 accepted from %s." % pc_img_ip)

            socket_pc_det, (pc_det_ip, pc_det_port) = socket_tcp.accept()  # 接收客户端的请求
            print("Destination2 accepted from %s." % pc_det_ip)

            socket_cam, (cam_ip, cam_port) = socket_tcp.accept()  # 接收客户端的请求
            print("Source accepted from %s." % cam_ip)
            self.access_flag = True
            return socket_pc_img, socket_pc_det, socket_cam
        elif self.source_type == 'phone':
            print("Waiting phone connect...")
            socket_pc_img, (pc_img_ip, pc_img_port) = socket_tcp.accept()  # 接收客户端的请求
            print("Destination1 accepted from %s." % pc_img_ip)

            socket_pc_det, (pc_det_ip, pc_det_port) = socket_tcp.accept()  # 接收客户端的请求
            print("Destination2 accepted from %s." % pc_det_ip)

            socket_phone, (phone_ip, phone_port) = socket_tcp.accept()  # 接收客户端的请求
            print("Source accepted from %s." % phone_ip)
            self.access_flag = True
            return socket_pc_img, socket_pc_det, socket_phone
            # socket_tcp.close()

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


def recieve_img(p_input):
    if imgsource.source_type == 'webcam':
        while True:
            # 接收图片
            length = imgsource.recv_all(socket_source, 16)  # 获得图片文件的长度,16 代表获取长度
            if not length:
                break
            stringData = imgsource.recv_all(socket_source, int(length))  # 根据获得的文件长度，获取图片文件
            data = np.frombuffer(stringData, np.uint8)  # 将获取到的字符流数据转换成1维数组
            decimg = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 将数组解码成图像
            if imgsource.send_flag1:
                p_input.send(decimg)
                imgsource.send_flag1 = False
            elif imgsource.send_flag:
                p_input.send(decimg)
                imgsource.send_flag = False
    elif imgsource.source_type == 'phone':
        while True:
            data1 = imgsource.recv_all(socket_source, 4)  # 接收开始标志
            if data1 is not None:
                start = int.from_bytes(data1, byteorder='little', signed=False)
                if start == 303030303:  # 收到开始标志，开始接收图像
                    data2 = imgsource.recv_all(socket_source, 4)  # 接收要接收的数据长度
                    image_length = int.from_bytes(data2, byteorder='little', signed=False)
                    image_bit = imgsource.recv_all(socket_source, image_length)
                    buff = np.frombuffer(image_bit, np.uint8)
                    img_decode = cv2.imdecode(buff, cv2.IMREAD_COLOR)
                    rotated = imgsource.rotate_img(img_decode, 90)  # 竖屏显示
                    mirror_img = cv2.flip(rotated, 1)  # 镜像
                    # 图像要求是RGB，所以此处需要转换图像的格式
                    RGB = cv2.cvtColor(mirror_img, cv2.COLOR_BGR2RGB)
                    RGB = cv2.resize(RGB, dsize=(480, 640))
                    if imgsource.send_flag1:
                        p_input.send(RGB)
                        imgsource.send_flag1 = False
                    elif imgsource.send_flag:
                        p_input.send(RGB)
                        imgsource.send_flag = False


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
        self.detections = []  # 用于存放结果

    def predict(self, img):
        rows, cols = img.shape[0], img.shape[1]
        res_arr = np.zeros((50, 6), dtype=np.float32)
        self.yolov5.Detect(self.c_point, c_int(rows), c_int(cols), img.ctypes.data_as(POINTER(c_ubyte)), res_arr)
        self.bbox_array = res_arr[~(res_arr == 0).all(1)]
        return self.bbox_array

    def visualize(self, img, bbox_array, conf_thres):
        self.detections = []
        for temp in bbox_array:
            if float(temp[5]) >= conf_thres:
                bbox = [int(temp[0]), int(temp[1]), int(temp[2]), int(temp[3])]  # xywh
                clas = int(temp[4])
                cls = self.name[clas]
                conf = float(temp[5])
                cxcy = [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2]
                self.detections.append({'class': cls, 'conf': conf, 'center_xy': cxcy, 'position': bbox})
                cv2.rectangle(img, (int(temp[0]), int(temp[1])), (int(temp[0] + temp[2]), int(temp[1] + temp[3])),
                              (105, 237, 249), 2)
                # img = cv2.putText(img, str(cls) + " " + str(round(conf, 2)), (int(temp[0]), int(temp[1]) - 5),
                #                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 237, 249), 1)
        return img, self.detections

    def free(self):
        self.yolov5.cuda_free(self.c_point)


def main(p_output, p_input2):
    imgsource.send_flag1 = True
    while True:
        image = p_output.recv()
        if image is not None:
            result = det.predict(image)  # 预测
            imgsource.send_flag = True  # 获取下一帧图像
            image, detections = det.visualize(image, result, 0.6)  # 画框，图像、识别结果、置信度阈值
            p_input2.send(image)
            if detections is not None:
                det_data = json.dumps(detections)
                socket_client_det.send(str.encode(str(len(det_data)).ljust(8)))
                # 发送数据
                socket_client_det.send(det_data)


def send_image(p_output2):
    # 压缩参数，后面cv2.imencode将会用到，对于jpeg来说，15代表图像质量，越高代表图像质量越好为 0-100，默认95
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    while True:
        img = p_output2.recv()
        if img is not None:
            # cv2.imencode将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输
            # '.jpg'表示将图片按照jpg格式编码。
            result, imgencode = cv2.imencode('.jpg', img, encode_param)
            # 建立矩阵
            data = np.array(imgencode)
            # 将numpy矩阵转换成字符形式，以便在网络中传输
            stringData = data.tobytes()
            # 先发送要发送的数据的长度
            # ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串
            socket_client.send(str.encode(str(len(stringData)).ljust(16)))
            # 发送数据
            socket_client.send(stringData)


if __name__ == "__main__":
    det = Detector(model_path=b"./myWeight/best_2400_L2phone.engine", dll_path="./myWeight/yolov5.dll")  # b'' is needed
    imgsource = ConnectImageSource('webcam', 30005)  # webcam; 30005 for phone port
    socket_client, socket_client_det, socket_source = imgsource.access()
    # Pipes
    p_output, p_input = Pipe()
    p_output2, p_input2 = Pipe()
    # lock = threading.RLock()
    th_list = []
    thread_0 = threading.Thread(target=recieve_img, args=[p_input])
    thread_1 = threading.Thread(target=main, args=[p_output, p_input2])
    thread_2 = threading.Thread(target=send_image, args=[p_output2])
    th_list.append(thread_0)
    th_list.append(thread_1)
    th_list.append(thread_2)
    for th in th_list:
        th.start()
        time.sleep(0.05)
