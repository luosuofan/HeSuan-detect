import base64
import json
import pickle
import socket
import threading
import time
from multiprocessing import Pipe

import cv2
import numpy as np
import pyttsx3
import qrcode

from myUtils.myUtils import Events, Example, Amplifier, cv2AddChineseText


class ConnectPhone():
    # 通过IP和port连接手机获取数据
    def __init__(self, port):
        self.port = port  # 30005 for phone port
        self.ip = self.get_host_ip()  # 主机作为AP热点的ip地址
        self.access_flag = False
        self.img = None
        self.imgRGB = None
        self.send_flag1 = False
        self.send_flag = False

    def access(self):
        print("Starting socket: TCP...")
        socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建socket

        print("TCP server listen @ %s:%d!" % (self.ip, self.port))
        host_addr = (self.ip, self.port)
        socket_tcp.bind(host_addr)  # 绑定主机的ip地址和端口号
        socket_tcp.listen(1)  # listen函数的参数是监听客户端的个数，这里只监听一个，即只允许与一个客户端创建连接

        print('waiting for connection...')
        socket_con, (client_ip, client_port) = socket_tcp.accept()  # 接收客户端的请求
        print("Connection accepted from %s." % client_ip)
        self.access_flag = True
        return socket_con

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


def genQRcode():
    IP = phone_connect.ip
    HOST = phone_connect.port
    text_QRcode = 'phonedets%' + IP + '%' + str(HOST)
    # 生成登录二维码
    qrcode.make(text_QRcode).save('./Access_QR.jpg')
    img_qr = cv2.imread("./Access_QR.jpg")
    img_qr = cv2.resize(img_qr, (480, 480))
    cv2.imshow("Access_QR", img_qr)
    cv2.moveWindow("Access_QR", 720, 300)  # 固定显示位置
    cv2.waitKey(10)
    while not phone_connect.access_flag:  # 等待用户接入，保持二维码窗口
        time.sleep(0.1)
    cv2.destroyWindow("Access_QR")  # 接入后销毁二维码窗口


def guide():
    pt = pyttsx3.init()
    if events.event_class == 0:
        # host = Linux('192.168.149.1', 'ubuntu', 'hiwonder')
        # host.connect()
        pt.say("图像显示界面处理中，请稍后...")
        pt.runAndWait()
        while phone_connect.access_flag is False:  # 等待图像接入
            time.sleep(0.5)
        pt.say("图像显示界面加载完成。请按示范操作")
        pt.runAndWait()
        stare_t = time.time()

        # 进入开始事件 event_num = 0
        pt.say("第一步，请对着镜头张嘴出现如左图采样区域提示，并确保口腔内图像清晰！")
        pt.runAndWait()
        my_time = 0
        pt.say("第二步，将咽拭子伸到咽喉方框采样区，在两侧扁桃体擦拭三到四次，中间擦拭二到三次")
        pt.runAndWait()
        while events.event_num == 0:  # 开始事件动作语音提示
            time.sleep(1)
            my_time += 1
            if my_time >= 5:
                my_time = 0
                if events.throat_flag:
                    pt.say("请用咽拭子在两侧扁桃体擦拭三到四次，中间擦拭二到三次")
                    pt.runAndWait()
                else:
                    pt.say("请用咽拭子从两侧伸至扁桃体处，并在各处来回旋转一到两圈")
                    pt.runAndWait()
        print('请按示范操作，将咽拭子伸到方框采样区，在两侧扁桃体擦拭三到四次，中间擦拭三到四次')

        # 进入采样事件 event_num = 1
        timeout_1 = 60  # 采样时超时限制
        my_time = 0
        while events.event_num == 1 and my_time < timeout_1:  # 处于采样事件且未超时，循环
            time.sleep(1)
            my_time += 1
        if my_time == timeout_1:
            events.event_num = 0  # 留用,超时退出重置事件流程
            pt.say("采样超时，未在规定时间内达到采样次数要求！！！")
            pt.runAndWait()
            print("采样超时，未在规定时间内达到采样次数要求！！！")
            return
        # host.send('cd ArmPi_PC_Software/', '')
        # host.send('python3 open_demo.py', '')
        pt.say("采样完成，请将拭子放入试管")
        pt.runAndWait()
        print('采样完成，请将拭子放入试管')
        example.gif1 = example.gif2
        example.gif1_index = example.gif2_index
        example.len_gif1 = example.len_gif2

        # 进入棉签追踪事件 event_num = 2
        time_alert = 0
        while events.event_num == 2:
            time.sleep(0.5)  # 每0.5秒判断一次棉签状态
            if events.swab_flag:
                time_alert = 0
            else:
                time_alert += 1
            if time_alert >= 4:  # 连续4次，即超过两秒未识别到棉签
                time_alert = 0
                pt.say("请勿遮挡棉签、或将棉签移出相机视野范围！！！")
                pt.runAndWait()

        # 进入棉签收集事件 event_num = 3
        pt.say("如提示，将棉签易断凹槽处卡于试管口并折断。")
        pt.runAndWait()
        time_alert = 0
        while events.event_num == 3:
            time.sleep(0.5)  # 每0.5秒判断一次棉签状态
            if events.intube_flag:
                time_alert = 0
            else:
                time_alert += 1
            if time_alert >= 4:  # 连续4次，即超过两秒未识别到棉签在试管中
                time_alert = 0
                pt.say("棉签进入试管后，折断过程中请勿抽出，以防掉落！！！")
                pt.runAndWait()
                pt.say("折断后请将棉签棒置于镜头前！！！")
                pt.runAndWait()

        # 进入结束事件 event_num = 4
        time_alert = 0
        while events.event_num == 4:
            time.sleep(0.5)  # 每0.5秒判断一次棉签状态
            if events.stick_flag:
                time_alert = 0
            else:
                time_alert += 1
            if time_alert >= 4:  # 连续4次，即超过两秒未识别到棉签棒
                time_alert = 0
                pt.say("请将折断后的棉签棒置于镜头前！！！")
                pt.runAndWait()

        end_t = time.time()
        use_t = int(end_t - stare_t)
        text_speak = '核酸采样已完成,采样流程用时%d秒！' % use_t
        pt.say(text_speak)
        pt.runAndWait()
        print('核酸采样已完成，采样流程用时%d秒！', use_t)
        # close_cap
        # host.send('python3 close_demo.py', '')
        # time.sleep(20)
        # host.close()
    elif events.event_class == 1:
        stare_t = time.time()  # 开始计时
        pt.say("请连接采样点提供WiFi后，扫描屏幕上二维码接入系统。")
        pt.runAndWait()
        # 等待图像接入
        time_alert = 0
        while phone_connect.access_flag is False:
            time.sleep(0.5)
            time_alert += 1
            if time_alert >= 20:  # 连续20次，即超过10秒未接入系统
                time_alert = 0
                pt.say("请确保成功连接采样点提供WIFI后，再扫描二维码接入系统！！！")
                pt.runAndWait()
        pt.say("图像显示界面加载完成。请按示范操作")
        pt.runAndWait()

        # 进入开始事件 event_ls_num = 0
        pt.say("第一步，请对着镜头张嘴出现如左图采样区域提示，并确保口腔内图像清晰！")
        pt.runAndWait()
        pt.say("第二步，将咽拭子伸到咽喉方框采样区，在两侧扁桃体擦拭三到四次，中间擦拭二到三次")
        pt.runAndWait()
        my_time = 0
        while events.event_ls_num == 0:  # 开始事件动作语音提示
            time.sleep(1)
            my_time += 1
            if my_time >= 5:
                my_time = 0
                if events.throat_flag:
                    pt.say("请用咽拭子在两侧扁桃体擦拭三到四次，中间擦拭二到三次")
                    pt.runAndWait()
                elif events.throat2_flag:
                    pt.say("请用咽拭子从两侧伸至扁桃体处，并在各处来回旋转一到两圈")
                    pt.runAndWait()
        print('请按示范操作，将咽拭子伸到方框采样区，在两侧扁桃体擦拭三到四次，中间擦拭三到四次')

        # 进入采样事件 event_ls_num = 1
        samplestare_t = time.time()
        timeout_1 = 60  # 采样时超时限制
        my_time = 0
        while events.event_ls_num == 1 and my_time < timeout_1:  # 处于采样事件且未超时，循环
            time.sleep(0.5)
            my_time += 0.5
            if events.sampling_flag:
                events.sampling_count += 1
                print("采样次数：", + int(events.sampling_count))
            elif events.sampling2_flag:
                events.sampling_count += 0.5
                print("采样次数：", + int(events.sampling_count))
        if my_time == timeout_1:
            events.event_ls_num = 0  # 留用,超时退出重置事件流程
            pt.say("采样超时，未在规定时间内达到采样次数要求！！！")
            pt.runAndWait()
            print("采样超时，未在规定时间内达到采样次数要求！！！")
            return
        sampleend_t = time.time()
        pt.say("采样完成后，请将拭子放入试管")
        pt.runAndWait()
        print('采样完成后，请将拭子放入试管')

        example.gif1 = example.gif2
        example.gif1_index = example.gif2_index
        example.len_gif1 = example.len_gif2

        # 进入结束事件 event_ls_num = 2
        time_alert = 0
        time_alert2 = 0
        while events.event_ls_num == 2:
            time.sleep(0.5)  # 每0.5秒判断一次棉签状态
            if events.stick_flag:
                time_alert = 0
            else:
                time_alert += 1
            if time_alert >= 10:  # 连续4次，即超过两秒未识别到棉签棒
                time_alert = 0
                pt.say("请将折断后的棉签棒置于镜头前！！！")
                pt.runAndWait()
            if events.sampling_flag:
                time_alert2 += 1
            else:
                time_alert2 -= 1
            if time_alert2 >= 4:  # 连续4次，即仍在进行上一步的采样动作
                time_alert2 = 0
                pt.say("采样完成，请按屏幕上左图示范动作，将棉签折入试管！！！")
                pt.runAndWait()

        end_t = time.time()
        sampleuse_t = int(sampleend_t - samplestare_t)
        alluse_t = int(end_t - stare_t)
        text_speak = '核酸采样已完成,采样用时%d秒！总流程用时%d秒！' % (sampleuse_t, alluse_t)
        pt.say(text_speak)
        pt.runAndWait()
        print('核酸采样已完成，采样流程用时%d秒！总流程用时%d秒！', sampleuse_t, alluse_t)


def recieve_datas(p_input):
    source = phone_connect.access()
    break_count = 0
    while True:
        data1 = phone_connect.recv_all(source, 4)  # 接收开始标志
        if data1 is not None:
            break_count = 0
            start = int.from_bytes(data1, byteorder='little', signed=False)
            if start == 303030303:  # 收到开始标志，开始接收图像
                data2 = phone_connect.recv_all(source, 4)  # 接收要接收的数据长度
                data_length = int.from_bytes(data2, byteorder='little', signed=False)

                data_bytes = phone_connect.recv_all(source, data_length)
                recData = eval(data_bytes)  # 反序列化
                # print(recData["results"])

                if phone_connect.send_flag1:
                    p_input.send(recData)
                    phone_connect.send_flag1 = False
                elif phone_connect.send_flag:
                    p_input.send(recData)
                    phone_connect.send_flag = False
        else:
            break_count += 1
            if break_count == 5:
                break  # 读取图片故障


def split_datas(data):
    # 分离图片数据
    image_bit = base64.b64decode(data['image'])  # Base64解码出图像bit
    buff = np.frombuffer(image_bit, np.uint8)
    img_decode = cv2.imdecode(buff, cv2.IMREAD_COLOR)
    # # 旋转竖屏显示
    img = phone_connect.rotate_img(img_decode, 0)
    # 镜像
    # img = cv2.flip(rotated, 1)
    # img = cv2.resize(img, dsize=(480, 640))

    # 分离检测结果
    detections = []
    if data['results'] != 101:
        for temp in data['results']:
            point = temp['position']
            bbox = [int(point[0]), int(point[1]), int(point[2]-point[0]), int(point[3]-point[1])]  # xywh
            cls = temp['class']
            conf = float(temp['conf'])
            cxcy = [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2]
            detections.append({'class': cls, 'conf': conf, 'center_xy': cxcy, 'position': bbox})
            print(detections)

    return img, detections


def main(p_output, p_input2):
    amplifier = Amplifier()
    phone_connect.send_flag1 = True
    while True:
        datas = p_output.recv()
        if datas is not None:
            image, detections = split_datas(datas)
            phone_connect.send_flag = True  # 获取下一帧图像
            events.get_result(detections)  # 传入检测结果
            events.process_event()  # 处理事件
            # 重点区域放大显示
            amplifier.set_detection(detections)
            amplifier.set_origin_img(image)
            amplified_img, flag = amplifier.get_img()
            frame = np.concatenate([image, amplified_img], axis=1)  # 原图和放大局部横向拼接
            p_input2.send(frame)


def show_image(p_output2):
    while True:
        img = p_output2.recv()
        if img is not None:
            frame = np.concatenate([example.gif1[example.gif1_index], img], axis=1)
            frame = cv2AddChineseText(frame, "示范区", (5, 30), (255, 0, 0), 30)
            cv2.imshow("Show", frame)
            cv2.moveWindow("Show", 300, 250)  # 固定显示位置
            example.gif1_index += 1
            if example.gif1_index == example.len_gif1:
                example.gif1_index = 0
            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyWindow("Show")
                break  # 按键退出


if __name__ == "__main__":
    phone_connect = ConnectPhone(30008)  # 使用30008端口连接手机
    events = Events(1)  # 事件检测，传入系统事件检测类别
    example = Example()  # 展示图像初始化
    # Pipes
    p_output, p_input = Pipe()
    p_output2, p_input2 = Pipe()
    # my para  init
    # lock = threading.RLock()
    th_list = []
    thread_0 = threading.Thread(target=genQRcode, args=[])
    thread_1 = threading.Thread(target=guide, args=[])
    thread_2 = threading.Thread(target=recieve_datas, args=[p_input])
    thread_3 = threading.Thread(target=main, args=[p_output, p_input2])
    thread_4 = threading.Thread(target=show_image, args=[p_output2])
    th_list.append(thread_0)
    th_list.append(thread_1)
    th_list.append(thread_2)
    th_list.append(thread_3)
    th_list.append(thread_4)
    for th in th_list:
        th.start()
