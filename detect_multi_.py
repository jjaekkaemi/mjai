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
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import subprocess
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh,clip_boxes, xywh2xyxy)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
import termios
@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference,
        vid_stride=1
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
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    count=1
    bs = 1  # batch_size
    # Dataloader
    if webcam:
        view_img = check_imshow(warn=True)
        
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # Process predictions
    crop_weights = "/opt/MVS/Samples/64/Python/GrabImage/yolov5_/yolov5/best-int8_edgetpu.tflite"
    crop_name = "data/coco128.yaml"
    crop_model = DetectMultiBackend(crop_weights, device=device, dnn=dnn, data=data, fp16=half)
            
    crop_stride, crop_names, crop_pt = crop_model.stride, crop_name, crop_model.pt
    print("crop_names", crop_names)
    crop_imgsz = check_img_size(imgsz, s=crop_stride)  # check image size
    crop_model.warmup(imgsz=(1 if crop_pt or crop_model.triton else bs, 3, *crop_imgsz))  # warmup
    for path, im, im0s, vid_cap, s in dataset:

        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        
        pred = model(im, augment=augment, visualize=visualize)
        
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        
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
            
            ob_xyxy_list = []
            ob_xywh_list = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    ob_xyxy_list.append(xyxy)
                    if xywh[2]<0.23 or xywh[3]<0.26 :
                        ob_xyxy_list = []
                        break
                #    '''
                #    * again pred
                #    '''
                for ob_xyxy in ob_xyxy_list:
                    ob_xyxy = torch.tensor(ob_xyxy).view(-1, 4)
                    b = xyxy2xywh(ob_xyxy)  # boxes
                    gain=1.02
                    pad=10
                    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
                    ob_xyxy = xywh2xyxy(b).long()
                    clip_boxes(ob_xyxy, imc.shape)
                    crop_img0s = imc[int(ob_xyxy[0, 1]):int(ob_xyxy[0, 3]), int(ob_xyxy[0, 0]):int(ob_xyxy[0, 2]), ::(1 if True else -1)]
   
                    
                    # Padded resize
                    crop_im = letterbox(crop_img0s, 416, stride=32, auto=True)[0]

                    # Convert
                    crop_im = crop_im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    crop_im = np.ascontiguousarray(crop_im)

                    crop_im = torch.from_numpy(crop_im).to(crop_model.device)
                    crop_im = crop_im.half() if crop_model.fp16 else crop_im.float()  # uint8 to fp16/32
                    crop_im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(crop_im.shape) == 3:
                        crop_im = crop_im[None]  # expand for batch dim
                    crop_pred = crop_model(crop_im, augment=augment, visualize=False)
                    crop_pred = non_max_suppression(crop_pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                    print("crop_model",crop_model)
                    for i, det in enumerate(crop_pred):  # per image
                        print("i :", i)

                        crop_im0 = crop_img0s.copy()

                        
                        gn = torch.tensor(crop_im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        crop_imc = crop_im0.copy() if save_crop else crop_im0  # for save_crop
                        annotator = Annotator(crop_im0, line_width=line_thickness, example=str(crop_names))
                        if len(det):
                            print(len(det))
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_boxes(crop_im.shape[2:], det[:, :4], crop_im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                
                            # Write results
                            for *xyxy, conf, cls in reversed(det):

                                c = int(cls)  # integer class
                                label = None if hide_labels else (crop_names[c] if hide_conf else f'{crop_names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                        crop_im0 = annotator.result()
                        cv2.imwrite(f"/home/dtriple/test/detect/test{count}.jpg", crop_im0)
                        count+=1
            
        
        #os.system("chmod 777 /opt/MVS/Samples/64/Python/GrabImage/yolov5/detect.py")
        #os.system("python3 /opt/MVS/Samples/64/Python/GrabImage/yolov5/detect.py --img 416 --weights oneboard_defect_labeling.pt --source /home/dtriple/test/detect")
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #     with open(f'{txt_path}.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # if save_img or save_crop or view_img:  # Add bbox to image
                    #     c = int(cls)  # integer class
                    #     label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    #     annotator.box_label(xyxy, label, color=colors(c, True))
                    # if save_crop:
                    #     print("[xyxy] ",xyxy)
                    #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        
      

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
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
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
