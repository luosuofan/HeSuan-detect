# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path
import math
import cv2
import torch
import torch.backends.cudnn as cudnn
from myUtils.Person import Person
import threading
import time
from myUtils.myUtils import *
import pyttsx3
from pyzbar import pyzbar
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, HuanLoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


def guide(person, event, max_num):
    # host = Linux('192.168.149.1', 'ubuntu', 'hiwonder')
    # host.connect()
    pt = pyttsx3.init()
    pt.say("图像显示界面处理中，请稍后...")
    pt.runAndWait()
    while person.uistate is False:      # 等待yolo初始化
        time.sleep(1)
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
    while events.event_num == 1 and my_time < timeout_1:     # 处于采样事件且未超时，循环
        time.sleep(1)
        my_time += 1
    if my_time == timeout_1:
        events.event_num = 0     # 留用,超时退出重置事件流程
        pt.say("采样超时，未在规定时间内达到采样次数要求！！！")
        pt.runAndWait()
        print("采样超时，未在规定时间内达到采样次数要求！！！")
        return
    # host.send('cd ArmPi_PC_Software/', '')
    # host.send('python3 open_demo.py', '')
    pt.say("采样完成，请将拭子放入试管")
    pt.runAndWait()
    print('采样完成，请将拭子放入试管')
    person.gif1 = person.gif2
    person.gif1_index = person.gif2_index
    person.len_gif1 = person.len_gif2

    # 进入棉签追踪事件 event_num = 2
    time_alert = 0
    while events.event_num == 2:
        time.sleep(0.5)       # 每0.5秒判断一次棉签状态
        if events.swab_flag:
            time_alert = 0
        else:
            time_alert += 1
        if time_alert >= 4:    # 连续4次，即超过两秒未识别到棉签
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


def scanqrcode():
    pt2 = pyttsx3.init()
    pt2.say("核酸采样系统已开启,请出示您的核酸码！")
    pt2.runAndWait()
    time_tip = 0
    # 1、读取二维码图片
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("错误!无法正常打开摄像头！！！")
        exit()
    while True:
        ret, qrcode = camera.read()
        if ret:
            # 2、解析二维码中的数据
            data = pyzbar.decode(qrcode)
            if data:
                # 3、在数据中解析出二维码的data信息
                text = data[0].data.decode('utf-8')
                print(text)
                if text == "0":
                    p1.qrstate = True
                    camera.release()
                    break
        else:
            print("获取帧失败！！！")
        time.sleep(0.5)  # 每0.5秒判断一次棉签状态
        if p1.qrstate:
            time_tip = 0
        else:
            time_tip += 1
        if time_tip >= 20:  # 连续4次，即超过两秒未识别到棉签
            time_tip = 0
            pt2.say("请出示您的核酸码！")
            pt2.runAndWait()


@torch.no_grad()
def run(weights=ROOT / 'yolov5l.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=6,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = HuanLoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        # im = cv2.flip(im, 1)   # 镜像翻转
        p1.uistate = True
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # 用于存放结果
        detections = []
        save_det = True
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                    if save_det:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                        xywh = [round(x) for x in xywh]
                        xywh = [xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2],
                                xywh[3]]  # 检测到目标位置，格式：（left，top，w，h)
                        cxcy = center(xywh)
                        cls = names[int(cls)]
                        conf = float(conf)
                        detections.append({'class': cls, 'conf': conf, 'center_xy': cxcy, 'position': xywh})

            # Stream results
            im0 = annotator.result()
            # 读取结果画形状
            for re in detections:
                if re['class'] == "throat" and re['conf'] >= 0.5:
                    a = int(max(re['position'][2], re['position'][3]) / 2)
                    b = int(min(re['position'][2], re['position'][3]) / 2)
                    cv2.ellipse(im0, re['center_xy'], (a, b), 0, 0, 360, (255, 255, 255), 1)  # 采样区域加椭圆
            events.get_result(detections)    # 传入检测结果
            events.process_event()           # 处理事件
            amplifier = Amplifier(detections, im0)
            amplify_img, flag = amplifier.get_img()
            if flag:
                p1.pre_amplify_img = amplify_img
            else:
                amplify_img = p1.pre_amplify_img
            if view_img:
                im0 = cv2.resize(im0, dsize=(640, 480))
                amplify_img = cv2.resize(amplify_img, dsize=(640, 480))
                frame = np.concatenate([p1.gif1[p1.gif1_index], im0, amplify_img], axis=1)
                # frame1 = np.concatenate([p1.gif1[p1.gif1_index], amplify_img], axis=1)
                # frame2 = cv2.copyMakeBorder(im0, 0, 0, 240, 240, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                # frame = np.concatenate([frame1, frame2], axis=0)
                cv2.putText(frame, "Example", (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(str(p), frame)
                cv2.moveWindow(str(p), -1920, 300)  # 固定显示位置
                p1.gif1_index += 1
                if p1.gif1_index == p1.len_gif1:
                    p1.gif1_index = 0
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def center(p):
    """
    返回中心坐标;
    :param p: [lx,ly,w,h]->[左上x坐标，左上y坐标]
    :return: [x,y]
    """
    return [p[0] + p[2] // 2, p[1] + p[3] // 2]


def point_distence(a, b):
    """
    两点间距离
    :param a:a点 (xa,ya)
    :param b: b点(xb,yb)
    :return: sqrt((xa-xb)**2 + (yb-ya)**2)
    """
    return math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'myWeight/best_2400_L2phone.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / "0", help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/covid19_7.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=6, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    # my para  init
    lock = threading.RLock()
    p1 = Person()
    events = Events()
    scanqrcode()
    MAX_NUM = 12  # 采样擦拭接触次数
    opt = parse_opt()
    th_list = []
    thread_1 = threading.Thread(target=guide, args=[p1, events, MAX_NUM])
    thread_2 = threading.Thread(target=main, args=[opt])
    th_list.append(thread_1)
    th_list.append(thread_2)

    for th in th_list:
        th.start()
