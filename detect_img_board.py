# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse, os, sys, torch, time, datetime
from pathlib import Path
import numpy as np
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

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
    ssd_path = "/media/dtriple/exFAT/mj_test"
    ori_path = "/home/dtriple/test/detect"
    now_datetime = datetime.datetime.now()
    ssd_datetime = f"{now_datetime.year}{change_time_format(now_datetime.month)}{change_time_format(now_datetime.day)}"
    save_path_dir = os.path.join(ssd_path, ssd_datetime)
    ori_path_dir = ""
    defect_path_dir = ""
    if os.path.exists(ssd_path)==False :
        save_path_dir = os.path.join(ori_path, ssd_datetime)
       
    if os.path.exists(save_path_dir) == False:
        os.makedirs(save_path_dir)
        ori_path_dir = os.path.join(save_path_dir, "ori")
        defect_path_dir = os.path.join(save_path_dir, "defect")
        if os.path.exists(ori_path_dir) == False:
            os.makedirs(ori_path_dir)
            os.makedirs(defect_path_dir)
    return ori_path_dir, defect_path_dir
 

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
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
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)

    dtime_ori, dtime_defect = getSavePathDir()

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    print(imgsz)
    # Dataloader
    bs = 1  # batch_size

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    crop_img_size = 416
    crop_iou_thres =  0.5
    
    crop_weights = "/opt/MVS/Samples/64/Python/GrabImage/yolov5_/defect-int8_edgetpu.tflite"
    crop_name = "/opt/MVS/Samples/64/Python/GrabImage/yolov5_/data/defect.yaml"
    crop_model = DetectMultiBackend(crop_weights, device=device, dnn=dnn, data=crop_name, fp16=half)

    crop_stride, crop_names, crop_pt = crop_model.stride, crop_model.names, crop_model.pt
    crop_imgsz = check_img_size(imgsz, s=crop_stride)  # check image size
    crop_model.warmup(imgsz=(1 if crop_pt or crop_model.triton else bs, 3, *crop_imgsz))  # warmup

    board_count = 0
    board_flag = False
    run_flag = True
    while run_flag:
        for path, im, im0s, vid_cap, s in dataset:
            dtime = datetime.datetime.now()
            dtime = f"{dtime.year}{change_time_format(dtime.month)}{change_time_format(dtime.day)}{change_time_format(dtime.hour)}{change_time_format(dtime.minute)}{change_time_format(dtime.second)}"
            dtime_defect = os.path.join(dtime_defect, dtime)
            dtime_ori = os.path.join(dtime_ori, dtime)
            
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                start_time = time.time()
                seen += 1

                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                ob_xyxy_list2 = []

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    xywh_count = 0
                    ob_xyxy_list = []
                    ob_xywh_list = []
                    for *xyxy, conf, cls in reversed(det):   
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

                        if (xywh[2]>=0.23 and xywh[3]>=0.25) and (xywh[0]>=0.30 and xywh[0]<=0.70):
                            xywh.append(xywh_count)
                            xywh_count += 1
                            ob_xyxy = torch.tensor(xyxy).view(-1, 4)
                            b = xyxy2xywh(ob_xyxy)
                            gain = 1.02
                            pad=10
                            b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
                            ob_xyxy = xywh2xyxy(b).long() 
                            clip_boxes(ob_xyxy, imc.shape)
                            crop_img0s = imc[int(ob_xyxy[0, 1]):int(ob_xyxy[0, 3]), int(ob_xyxy[0, 0]):int(ob_xyxy[0, 2]), ::(1 if True else -1)]
                            ob_xyxy_list.append(crop_img0s)
                            ob_xywh_list.append(xywh)

                    defect_flag = False
                    if len(ob_xyxy_list)>0 and len(ob_xyxy_list)%2 == 0 :
                        check_list = []
                        for ob_xywh in range(len(ob_xywh_list)):
                            if ob_xywh not in check_list:
                                two_board_check = True
                                for check_xywh in range(ob_xywh+1, len(ob_xywh_list)):
                                    if check_xywh not in check_list :
                                        if 0.30<ob_xywh_list[ob_xywh][0] and ob_xywh_list[ob_xywh][0]<0.45:
                                            if 0.55<ob_xywh_list[check_xywh][0] and ob_xywh_list[check_xywh][0]<0.70:
                                                check_list.append(ob_xywh)
                                                check_list.append(check_xywh)
                                                two_board_check = True
                                                break
                                            else:
                                                two_board_check = False
                                        elif 0.55<ob_xywh_list[ob_xywh][0] and ob_xywh_list[ob_xywh][0]<0.70:
                                            if 0.30<ob_xywh_list[check_xywh][0] and ob_xywh_list[check_xywh][0]<0.45:
                                                check_list.append(ob_xywh)
                                                check_list.append(check_xywh)
                                                two_board_check = True
                                                break
                                            else :
                                                two_board_check = False
                                        else:
                                            two_board_check = False
                                            
                                if two_board_check == False:
                                    ob_xyxy_list = []
                                    ob_xywh_list = []
                                    ob_xyxy_list2 = []
                                    board_flag = False
                                    break
                        if len(ob_xyxy_list)>0 and board_flag == False:
                            board_count += len(ob_xyxy_list)
                            board_flag = True
                            
                        ob_xywh_list.sort(key=lambda x:x[1])
                        for xy in range(len(ob_xyxy_list)//2):
                            xy_i = xy*2
                            if ob_xywh_list[xy_i][0]<ob_xywh_list[xy_i+1][0]:
                                ob_xyxy_list2.append(ob_xyxy_list[ob_xywh_list[xy_i][4]])
                                ob_xyxy_list2.append(ob_xyxy_list[ob_xywh_list[xy_i+1][4]])
                            else:
                                ob_xyxy_list2.append(ob_xyxy_list[ob_xywh_list[xy_i+1][4]])
                                ob_xyxy_list2.append(ob_xyxy_list[ob_xywh_list[xy_i][4]])
                    else: 
                        ob_xyxy_list = []
                        ob_xywh_list = []
                        ob_xyxy_list2 = []
                        board_flag = False
                    result_list = []
                    for ob_xyxy in ob_xyxy_list2:

                        crop_im__ = letterbox__(ob_xyxy, crop_img_size, stride=32, auto=True)[0]
                        # # Convert
                        crop_im = crop_im__.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                        crop_im = np.ascontiguousarray(crop_im)

                        crop_im = torch.from_numpy(crop_im).to(crop_model.device)
                        crop_im = crop_im.half() if crop_model.fp16 else crop_im.float()  # uint8 to fp16/32
                        crop_im /= 255  # 0 - 255 to 0.0 - 1.0
                        if len(crop_im.shape) == 3:
                            crop_im = crop_im[None]  # expand for batch dim
                            
                        crop_pred = crop_model(crop_im, augment=augment, visualize=False)
                        crop_pred = non_max_suppression(crop_pred, crop_iou_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                            
                        for i, det in enumerate(crop_pred):
                            crop_im0 = ob_xyxy.copy()
                            gn_defect = torch.tensor(crop_im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                                
                            annotator = Annotator(crop_im0, line_width=line_thickness, example=str(crop_names))
                            file_s = "\n============================================================\n"
                            if len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_boxes(crop_im.shape[2:], det[:, :4], crop_im0.shape).round()

                                # Print results
                                for c in det[:, -1].unique():
                                    n = (det[:, -1] == c).sum()  # detections per class
                                    
                                # Write results
                                for *xyxy, conf, cls in reversed(det):
                                    xywh_defect = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn_defect).view(-1).tolist()
                                    c = int(cls)  # integer class
                                    file_s += f"[defect] {crop_names[c]} {conf:.2f} {xywh_defect}\n"
                                    if c != 15 :
                                        if c == 1 :
                                            if xywh_defect[2]>0.035: 
                                                defect_flag = True
                                                label = None if hide_labels else (crop_names[c] if hide_conf else f'{crop_names[c]} {conf:.2f}')
                                                annotator.box_label(xyxy, label, color=colors(c, True))
                                        else :
                                            defect_flag = True
                                            label = None if hide_labels else (crop_names[c] if hide_conf else f'{crop_names[c]} {conf:.2f}')
                                            annotator.box_label(xyxy, label, color=colors(c, True))
                            crop_im0 = annotator.result()
                            crop_im0 = cv2.resize(crop_im0, (700, 560))
                            result_list.append(crop_im0)            
                    if len(result_list)%2 == 0 and defect_flag:
                        img_list = []
                        for i in range(len(result_list)//2):
                            img_list.append([result_list[i*2], result_list[i*2+1]])
                        save_img = cv2.vconcat([cv2.hconcat(img) for img in img_list])

                        file_s += f"===================={dtime_defect}.jpg====================\n"
                        #defect_file = open ("/home/dtriple/test/230314.txt", "a")
                        #defect_file.write(file_s)
                        #defect_file.close()
                        #cv2.imwrite(dtime_defect+".jpg", save_img)
                        save_img = cv2.resize(save_img, (save_img.shape[0]//2, save_img.shape[1]//2))
                        cv2.imshow("test", save_img)
                        cv2.waitKey(1) 
                            
                        # write_file = os.path.join(save_img_dir+"/defect", f"{p.stem}{board_count}")
                        # cv2.imwrite(f"{write_file}.jpg", crop_im0)

                
                # print(f"board: {p.stem}",board_count)
            
            # Print time (inference-only)
            # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        
        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    def resize_and_pad(image, desired_size):
        old_size = image.shape[:2] 
        ratio = float(desired_size/max(old_size))
        new_size = tuple([int(x*ratio) for x in old_size])
        
        # new_size should be in (width, height) format
        
        image = cv2.resize(image, (new_size[1], new_size[0]))
        
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        
        pad = (delta_w, delta_h)
        
        color = [114, 114, 114]
        new_im = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT,
            value=color)
            
        return new_im, pad

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
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
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
