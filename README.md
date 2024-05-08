 - add HuanLoadStreams,

 
在cap.set(cv2.CAP_PROP_FRAME_WIDTH, values)
方法中可以设设置载入长宽，**对应对detect.py中，if webcam 的LoadStreams**
note:
1. 2k usb_cam , use HuanLoadStreams
2. video stream of phone(640 * 480), use LoadStreams(original yolov5 Loading)
