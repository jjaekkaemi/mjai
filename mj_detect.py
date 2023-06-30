import argparse, os, sys, torch, time, datetime
from pathlib import Path
import numpy as np
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, Profile, check_img_size, check_requirements, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh, clip_boxes, xywh2xyxy)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox__

def change_time_format(str_time):
	if len(str(str_time))==2 :
		return str(str_time)
	else :
		return '0'+str(str_time)
def getSavePathDir():
    ssd_path = "/media/user/exFAT/mj_test"
    now_datetime = datetime.datetime.now()
    ssd_datetime = f"{now_datetime.year}{change_time_format(now_datetime.month)}{change_time_format(now_datetime.day)}"
    save_path_dir = os.path.join(ssd_path, ssd_datetime)
    ori_path_dir = ""
    defect_path_dir = ""

    if os.path.exists(save_path_dir) == False:
        os.makedirs(save_path_dir)
    ori_path_dir = os.path.join(save_path_dir, "ori")
    defect_path_dir = os.path.join(save_path_dir, "defect")
    if os.path.exists(ori_path_dir) == False:
        os.makedirs(ori_path_dir)
        os.makedirs(defect_path_dir)
    return ori_path_dir, defect_path_dir

class EdgeTpuModel():
    def __init__(self,  weights, class_name, imgsz = [416,416], conf_thres = 0.5, iou_thres = 0.45, classes = None):
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.weights = weights
        self.class_name = class_name
        self.classes = classes
        device=''
        dnn=False
        half=False

        device = select_device(device)
        self.model = DetectMultiBackend(self.weights, device=device, dnn=dnn, data=self.class_name, fp16=half)

        bs = 1  # batch_size
        self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else bs, 3, *self.imgsz))  # warmup
        self.im = None
        self.pred = None
        self.frame = None
        self.im0 = None
        self.predict_list = []
    def set_classes(self, del_classes):
        self.classes.remove(del_classes)
    def add_classes(self, add_classes):
        self.classes.append(add_classes)

    def img_processing(self, frame):
        self.frame = frame
        im = letterbox__(frame, self.imgsz[0], 32, True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous


        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.im = im

    def inference(self):
        pred = self.model(self.im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, False, max_det=1000)
        self.pred = pred

    def img_crop(self, xyxy, imc):
        gain = 1.02
        pad=10

        xyxy = torch.tensor(xyxy).view(-1, 4)
        b = xyxy2xywh(xyxy)
        
        b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
        xyxy = xywh2xyxy(b).long() 
        clip_boxes(xyxy, imc.shape)
        crop_img0s = imc[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if True else -1)]
        return crop_img0s
    
    def process_predictions(self, pred, im0s, im):
        self.predict_list = []

        for i, det in enumerate(self.pred):
            self.im0=self.frame.copy()
            gn = torch.tensor(self.im0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_boxes(self.im.shape[2:], det[:, :4], self.im0.shape).round()
                detection_list = []
                for *xyxy, conf, cls in reversed(det): 
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    detection_list.append({"xyxy":xyxy, "xywh":xywh,"conf":conf, "cls":cls })
                self.predict_list.append(detection_list)

