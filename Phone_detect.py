import threading
from multiprocessing import Pipe
from myUtils.myUtils import *
import qrcode
import pyttsx3
import time


def guide():
    pt = pyttsx3.init()
    if events.event_class == 0:
        # host = Linux('192.168.149.1', 'ubuntu', 'hiwonder')
        # host.connect()
        pt.say("图像显示界面处理中，请稍后...")
        pt.runAndWait()
        while imgsource.access_flag is False:  # 等待图像接入
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
        if imgsource.source_type == 'phone':
            pt.say("请连接采样点提供WiFi后，扫描屏幕左上角二维码接入系统。")
            pt.runAndWait()
            # 等待图像接入
            time_alert = 0
            while imgsource.access_flag is False:
                time.sleep(0.5)
                time_alert += 1
                if time_alert >= 20:  # 连续20次，即超过10秒未接入系统
                    time_alert = 0
                    pt.say("请确保成功连接采样点提供WIFI后，再扫描二维码接入系统！！！")
                    pt.runAndWait()
        elif imgsource.source_type == 'webcam':
            while imgsource.access_flag is False:
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

        end_t = time.time()
        sampleuse_t = int(sampleend_t - samplestare_t)
        alluse_t = int(end_t - stare_t)
        text_speak = '核酸采样已完成,采样用时%d秒！总流程用时%d秒！' % (sampleuse_t, alluse_t)
        pt.say(text_speak)
        pt.runAndWait()
        print('核酸采样已完成，采样流程用时%d秒！总流程用时%d秒！', sampleuse_t, alluse_t)


def recieve_img(p_input):
    imgsource.access(p_input)


def main(p_output, p_input2):
    amplifier = Amplifier()
    imgsource.send_flag1 = True
    while True:
        image = p_output.recv()
        if image is not None:
            result = det.predict(image)  # 预测
            imgsource.send_flag = True  # 获取下一帧图像
            image, detections = det.visualize(image, result, 0.6)  # 画框，图像、识别结果、置信度阈值
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
            cv2.moveWindow("Show", -1820, 250)  # 固定显示位置
            example.gif1_index += 1
            if example.gif1_index == example.len_gif1:
                example.gif1_index = 0
            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break


def genQRcode():
    if imgsource.source_type == 'phone':
        IP = imgsource.ip
        HOST = imgsource.source
        text_QRcode = 'covid%' + IP + '%' + str(HOST)
        # 生成登录二维码
        qrcode.make(text_QRcode).save('./Login_QR.jpg')
        img_qr = cv2.imread("./Login_QR.jpg")
        img_qr = cv2.resize(img_qr, (480, 480))
        cv2.imshow("Login_QR", img_qr)
        cv2.moveWindow("Login_QR", 0, 0)  # 固定显示位置
        cv2.waitKey(10)
        while not imgsource.access_flag:  # 等待用户接入，保持二维码窗口
            sleep(0.1)
        cv2.destroyWindow("Login_QR")  # 接入后销毁二维码窗口


if __name__ == "__main__":
    det = Detector(model_path=b"./myWeight/best_2400_L2phone.engine", dll_path="./myWeight/yolov5.dll")  # b'' is needed
    events = Events(1)  # 事件检测，传入系统事件检测类别
    # globalvar = GlobalVar(20)

    imgsource = ConnectImageSource('phone', 30005)  # 0,1.. or url for webcam; 30005 for phone port
    example = Example()  # 展示图像初始化
    # Pipes
    p_output, p_input = Pipe()
    p_output2, p_input2 = Pipe()
    # my para  init
    # lock = threading.RLock()
    th_list = []
    thread_0 = threading.Thread(target=genQRcode, args=[])
    thread_1 = threading.Thread(target=guide, args=[])
    thread_2 = threading.Thread(target=recieve_img, args=[p_input])
    thread_3 = threading.Thread(target=main, args=[p_output, p_input2])
    thread_4 = threading.Thread(target=show_image, args=[p_output2])
    th_list.append(thread_0)
    th_list.append(thread_1)
    th_list.append(thread_2)
    th_list.append(thread_3)
    th_list.append(thread_4)
    for th in th_list:
        th.start()


