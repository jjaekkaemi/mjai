import os
import sys
import argparse
import logging
import time
from pathlib import Path
import glob
import json

import numpy as np
from tqdm import tqdm
import cv2
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from edgetpumodel import EdgeTPUModel
from utils import resize_and_pad, get_image_tensor, save_one_json, coco80_to_coco91_class

import sys
import threading
import termios
import time
import cv2
import numpy as np

from ctypes import *

sys.path.append("../../MvImport")
from MvCameraControl_class import *
g_bExit = False
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
board_model = "defect-int8_416_edgetpu.tflite"
board_names = "data/defect.yaml"
def image_control(data, stFrameInfo):
    data = data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
    
    image_data = cv2.cvtColor(data, cv2.COLOR_BAYER_GR2RGB)
    image_data = cv2.resize(image_data, (640, 640))
    return image_data
    
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser("EdgeTPU test runner")
    parser.add_argument("--model", "-m", help="weights file", required=True)
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="model confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--names", type=str, default='data/coco.yaml', help="Names file")
    parser.add_argument("--image", "-i", type=str, help="Image file to run detection on")
    parser.add_argument("--device", type=int, default=1, help="Image capture device to run live detection")
    parser.add_argument("--stream", action='store_true', help="Process a stream")
    parser.add_argument("--bench_coco", action='store_true', help="Process a stream")
    parser.add_argument("--coco_path", type=str, help="Path to COCO 2017 Val folder")
    parser.add_argument("--quiet","-q", action='store_true', help="Disable logging (except errors)")
        
    args = parser.parse_args()
    
    if args.quiet:
        logging.disable(logging.CRITICAL)
        logger.disabled = True
    
    if args.stream and args.image:
        logger.error("Please select either an input image or a stream")
        exit(1)
    
    model = EdgeTPUModel(args.model, args.names, conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh)
    input_size = model.get_image_size()
    print(input_size)
    x = (255*np.random.random((3,*input_size))).astype(np.uint8)
    model.forward(x)

    #defect_model = EdgeTPUModel(board_model, board_names, conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh)
    #defect_model.forward(x)
    conf_thresh = 0.25
    iou_thresh = 0.45
    classes = None
    agnostic_nms = False
    max_det = 1000

    #elif args.stream:
    if args.stream:
        logger.info("Opening stream on device: {}".format(args.device))
        
        SDKVersion = MvCamera.MV_CC_GetSDKVersion()
        print ("SDKVersion[0x%x]" % SDKVersion)
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print ("enum devices fail! ret[0x%x]" % ret)
            sys.exit()
        if deviceList.nDeviceNum == 0:
            print ("find no device!")
            sys.exit()
        print ("Find %d devices!" % deviceList.nDeviceNum)
        
        cam = MvCamera()
        nConnectionNum = 0
        stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print ("create handle fail! ret[0x%x]" % ret)
            sys.exit()
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print ("open device fail! ret[0x%x]" % ret)
            sys.exit()
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
                if ret != 0:
                    print ("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print ("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print ("set trigger mode fail! ret[0x%x]" % ret)
            sys.exit()
        stParam =  MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        
        
        ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print ("get payload size fail! ret[0x%x]" % ret)
            sys.exit()
        nPayloadSize = stParam.nCurValue
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            print ("start grabbing fail! ret[0x%x]" % ret)
            sys.exit()

        data_buf = (c_ubyte * nPayloadSize)()
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 1000)
        if ret != 0:
            sys.exit()
        # cap = cv2.VideoCapture(args.device)
        #cap = cv2.VideoCapture(1)
        frameWidth = int(stFrameInfo.nWidth)
        frameHeight = int(stFrameInfo.nHeight)
        frame_size = (frameWidth, frameHeight)
        print(frame_size)

        k=0 
        while True:
          try:

            ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 1000)
            if ret == 0:
                print ("get one frame: Width[%d], Height[%d], PixelType[0x%d], nFrameNum[%d]"  % (stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.enPixelType,stFrameInfo.nFrameNum))
                data = np.frombuffer(data_buf, count=int(stFrameInfo.nFrameLen), dtype=np.uint8)
                frame = image_control(data=data, stFrameInfo=stFrameInfo)
            else:
                print ("no data[0x%x]" % ret)
            if g_bExit == True:
                break

            full_image, net_image, pad = get_image_tensor(frame, input_size[0])
            pred = model.forward(net_image)
            print(full_image.shape)
            model.process_predictions(pred[0], full_image, pad)
            
            tinference, tnms = model.get_last_inference_time()
            cv2.imshow('result', frame)
            
            
            logger.info("Frame done in {}".format(tinference+tnms))

            key = cv2.waitKey(1) & 0xFF
            if(k==27):
               break

          except KeyboardInterrupt:
            g_bExit = True
            ret = cam.MV_CC_StopGrabbing()
            if ret != 0:
                print ("stop grabbing fail! ret[0x%x]" % ret)
                del data_buf
                sys.exit()
            ret = cam.MV_CC_CloseDevice()
            if ret != 0:
                print ("close deivce fail! ret[0x%x]" % ret)
                del data_buf
                sys.exit()

            ret = cam.MV_CC_DestroyHandle()
            if ret != 0:
                print ("destroy handle fail! ret[0x%x]" % ret)
                del data_buf
                sys.exit()

            del data_buf
            break
          
        # cam.release()
    elif args.image:
        files = []
        path = args.image
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        for i in images:
            frame = cv2.imread(i)
            assert frame is not None, f'Image Not Found {path}'
            full_image, net_image, pad = get_image_tensor(frame, input_size[0])
            pred = model.forward(net_image)
            print(pred)
            crop_im = model.process_predictions(pred[0], full_image, pad)
            # tinference, tnms = model.get_last_inference_time()
            # total_time = 0
            # for c in crop_im :
            #     defect_full_image, defect_net_image, defect_pad = get_image_tensor(c, input_size[0])
            #     defect_pred = defect_model.forward(defect_net_image)
            #     defect_full_image = np.ascontiguousarray(defect_full_image)
            #     defect_model.process_predictions(defect_pred[0], defect_full_image, defect_pad)
            
            #     defect_tinference, defect_tnms = defect_model.get_last_inference_time()
            #     total_time = defect_tinference+defect_tnms+total_time
            # if total_time!=0:
            #     logger.info("Frame done in {}".format(tinference+tnms+total_time))