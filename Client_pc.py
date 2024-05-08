import json
import threading
import socket
import cv2
import numpy
import time
import sys
import qrcode
import pyttsx3
from myUtils.myUtils import *


def socket_con(ip, port):
    address = (ip, port)
    try:
        # socket.AF_INET：服务器之间网络通信
        # socket.SOCK_STREAM：流式socket , for TCP
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 开启连接
        sock.connect(address)
        return sock
    except socket.error as msg:
        print(msg)
        sys.exit(1)


def get_result():
    socket_get_img = socket_con(globalval.server_ip, globalval.server_host)
    print('get img!')

    def recvall(s, count):
        buf = b''  # buf是一个byte类型
        while count:
            # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
            newbuf = s.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    amplifier = Amplifier()
    while True:
        # 接收图片
        length = recvall(socket_get_img, 16)  # 获得图片文件的长度,16 代表获取长度
        if not length:
            break
        globalval.access_img_flag = True
        stringdata = recvall(socket_get_img, int(length))  # 根据获得的文件长度，获取图片文件
        data = numpy.frombuffer(stringdata, numpy.uint8)  # 将获取到的字符流数据转换成1维数组
        decimg = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 将数组解码成图像

        events.get_result(globalval.detections)  # 传入检测结果
        events.process_event()  # 处理事件
        # 重点区域放大显示
        amplifier.set_detection(globalval.detections)
        amplifier.set_origin_img(decimg)
        amplified_img, flag = amplifier.get_img()
        frame = np.concatenate([decimg, amplified_img], axis=1)  # 原图和放大局部横向拼接

        frame = np.concatenate([example.gif1[example.gif1_index], frame], axis=1)  # 拼接展示图
        frame = cv2AddChineseText(frame, "示范区", (5, 30), (255, 0, 0), 30)
        cv2.imshow("Show", frame)
        cv2.moveWindow("Show", -1820, 250)  # 固定显示位置
        example.gif1_index += 1
        if example.gif1_index == example.len_gif1:
            example.gif1_index = 0
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
    socket_get_img.close()


def get_result_dets():
    socket_get_dets = socket_con(globalval.server_ip, globalval.server_host)
    print('get dets!')

    def recvall(s, count):
        buf = b''  # buf是一个byte类型
        while count:
            # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
            newbuf = s.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    while True:
        # 接收图片
        length = recvall(socket_get_dets, 8)  # 获得图片文件的长度,8 代表获取长度
        if not length:
            break
        globalval.access_img_flag = True
        stringdata = recvall(socket_get_dets, int(length))  # 根据获得的文件长度，获取图片文件
        json_string = json.loads(stringdata)
        globalval.detections = json_string
        # print(json_string)
    socket_get_dets.close()


def sendvideo():
    address = ('10.23.14.240', 30005)
    # address = ("127.0.0.1", 30005)
    try:
        sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_send.connect(address)
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    print('send img!')

    # 建立图像读取对象
    capture = cv2.VideoCapture(1)
    # 压缩参数，后面cv2.imencode将会用到，对于jpeg来说，15代表图像质量，越高代表图像质量越好为 0-100，默认95
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

    while True:
        # 读取一帧图像，读取成功:ret=1 frame=读取到的一帧图像；读取失败:ret=0
        ret, frame = capture.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.imencode将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输
            # '.jpg'表示将图片按照jpg格式编码。
            result, imgencode = cv2.imencode('.jpg', img, encode_param)
            # 建立矩阵
            data = numpy.array(imgencode)
            # 将numpy矩阵转换成字符形式，以便在网络中传输
            stringdata = data.tobytes()

            # 先发送要发送的数据的长度
            # ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串
            sock_send.send(str.encode(str(len(stringdata)).ljust(16)))
            # 发送数据
            sock_send.send(stringdata)


def genQRcode():
    if globalval.source_type == 'phone':
        ip = globalval.server_ip
        host = globalval.server_host
        text_qrcode = 'covid%' + ip + '%' + str(host)
        # 生成登录二维码
        qrcode.make(text_qrcode).save('./Login_QR.jpg')
        img_qr = cv2.imread("./Login_QR.jpg")
        img_qr = cv2.resize(img_qr, (480, 480))
        cv2.imshow("Login_QR", img_qr)
        # cv2.moveWindow("Login_QR", -1200, 300)  # 固定显示位置
        cv2.waitKey(10)
        while not globalval.access_img_flag:  # 等待用户接入，保持二维码窗口
            time.sleep(0.1)
        cv2.destroyWindow("Login_QR")  # 接入后销毁二维码窗口


class GlobalVar:
    def __init__(self, source_type, server_ip, server_host):
        self.source_type = source_type
        self.server_ip = server_ip
        self.server_host = server_host
        self.access_img_flag = False
        self.sample_state = False
        self.detections = []

    def re_initstate(self):
        self.access_img_flag = False
        self.sample_state = False
        self.detections = []


def guide():
    pt = pyttsx3.init()
    if events.event_class == 0:
        # host = Linux('192.168.149.1', 'ubuntu', 'hiwonder')
        # host.connect()
        pt.say("图像显示界面处理中，请稍后...")
        pt.runAndWait()
        while globalval.access_img_flag is False:  # 等待图像接入
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
        if globalval.source_type == 'phone':
            pt.say("请连接采样点提供WiFi后，扫描屏幕上二维码接入系统。")
            pt.runAndWait()
            # 等待图像接入
            time_alert = 0
            while globalval.access_img_flag is False:
                time.sleep(0.5)
                time_alert += 1
                if time_alert >= 20:  # 连续20次，即超过10秒未接入系统
                    time_alert = 0
                    pt.say("请确保成功连接采样点提供WIFI后，再扫描二维码接入系统！！！")
                    pt.runAndWait()
        elif globalval.source_type == 'webcam':
            while globalval.access_img_flag is False:
                time.sleep(0.5)
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

        globalval.sample_state = True  # 检测流程完成
        end_t = time.time()
        sampleuse_t = int(sampleend_t - samplestare_t)
        alluse_t = int(end_t - stare_t)
        text_speak = '核酸采样已完成,采样用时%d秒！总流程用时%d秒！' % (sampleuse_t, alluse_t)
        pt.say(text_speak)
        pt.runAndWait()
        print('核酸采样已完成，采样流程用时%d秒！总流程用时%d秒！', sampleuse_t, alluse_t)


if __name__ == '__main__':
    globalval = GlobalVar('phone', '10.23.14.240', 30005)
    events = Events(1)  # 事件检测，传入系统事件检测类别
    example = Example()  # 展示图像初始化
    th_list = []
    thread_0 = threading.Thread(target=get_result, args=[])
    thread_1 = threading.Thread(target=get_result_dets, args=[])
    th_list.append(thread_0)
    th_list.append(thread_1)
    if globalval.source_type == 'webcam':
        thread_2 = threading.Thread(target=sendvideo, args=[])
        th_list.append(thread_2)
    elif globalval.source_type == 'phone':
        thread_2 = threading.Thread(target=genQRcode, args=[])
        th_list.append(thread_2)
    thread_3 = threading.Thread(target=guide, args=[])
    th_list.append(thread_3)
    for th in th_list:
        th.start()
        time.sleep(0.05)
