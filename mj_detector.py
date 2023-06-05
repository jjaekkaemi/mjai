
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from qt_style import TitleLabel, TitleCombox
import datetime
from mj_detect import EdgeTpuModel

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
sys.path.append("../../MvImport")


import requests, json
from MvCameraControl_class import *
import cv2
from database import SQLDatabase

EDGE_CHECK_SIZE = [0, 80, 80]


def change_time_format(str_time):
	if len(str(str_time))==2 :
		return str(str_time)
	else :
		return '0'+str(str_time)

def post_inspection(inspection_json):
    try:
        url = f"https://mj.d-triple.com/api/mes/v1/external/inspection"
        
        headers = {"Content-Type": "application/json"}
        print(inspection_json)
        response = requests.post(url, headers=headers, json=inspection_json)
        return response.status_code
    except:
        return 400
        print("error")
        
def image_control(data, stFrameInfo):
    data = data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
    
    image_data = cv2.cvtColor(data, cv2.COLOR_BAYER_GR2RGB)
    
    return image_data
def initMvCamera():
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
    return cam, data_buf, stFrameInfo, nPayloadSize


class BoardDefectDetect(QThread):
    def __init__(self, workorder_quantity, inspection_quantity, inspection_percent, bad_quantity, 
                 bad_type, normal_quantity,bad_img_label, workorder, workorder_item, inspector_name, 
                 inspection_date, inspection_start_time, inspection_stop_time, defect_img_label, stackedWidget):
        super().__init__()
        # cam, data_buf, stFrameInfo, nPayloadSize  = initMvCamera()
        board_weights = "/opt/MVS/Samples/64/Python/GrabImage/yolov5_/twoboard_detect-int8_edgetpu.tflite"
        board_name = "/opt/MVS/Samples/64/Python/GrabImage/yolov5_/data/board.yaml"
        defect_weights = "/opt/MVS/Samples/64/Python/GrabImage/yolov5_/defect_416_s_2-int8_edgetpu.tflite"
        defect_name = "/opt/MVS/Samples/64/Python/GrabImage/yolov5_/data/defect.yaml"
        self.board_model = EdgeTpuModel(board_weights, board_name, conf_thres=0.7)
        self.defect_model = EdgeTpuModel(defect_weights, defect_name)

        self.working = False
        # self.cam = cam
        # self.data_buf = data_buf
        # self.stFrameInfo = stFrameInfo
        # self.nPayloadSize = nPayloadSize
        self.line_thickness = 6
        
        self.defect_flag = False
        self.defect_type_flag = [False,False,False,False,False,False]
        self.defect_board_flag = [False,False,False,False,False,False]
        self.defect_count_list = [0,0,0,0,0,0]
        self.defect_alarm = False
        self.board_count = 0 
        self.defect_count = 0 
        self.defect_type_label = ""
        self.board_flag = False
        self.before_board_flag = False
        self.test_count = 0
        self.defect_type = [0, 0, 0, 0, 0, 0, 0,1, 1,0, 2,1,0, 0, 0, 3, 1, 3, 0, 3, 3, 3, 1]
        self.defect_type_count = [0,0,0]
        self.defect_type_list = ["미납","리드미삽", "쇼트", ""]
        self.defect_show_flag = False
        self.defect_show_list = []
        self.defect_img_label = defect_img_label
        self.stackedWidget = stackedWidget


        self.workorder_quantity = workorder_quantity
        self.inspection_quantity = inspection_quantity
        self.inspection_percent = inspection_percent
        self.bad_quantity = bad_quantity
        self.bad_type = bad_type
        self.normal_quantity = normal_quantity
        self.bad_img_label = bad_img_label
        self.workorder = workorder
        self.workorder_item = workorder_item
        self.inspection_date = inspection_date
        self.start_date = inspection_start_time
        self.end_date = inspection_stop_time
        self.inspector_name = inspector_name

        self.sqldatabase = SQLDatabase()
        
        self.sqldatabase.check_post_data()
        self.today_inspection = self.sqldatabase.check_today_table(self.get_inspection_json(), self.workorder.get_current_text())

        self.start_date.setText(str(self.today_inspection[1]))
        self.inspection_date.change_label(str(self.today_inspection[5]))
        self.workorder_quantity.change_label(str(self.today_inspection[7]))
        self.inspection_quantity.change_label(str(self.today_inspection[8]))
        self.inspection_percent.change_label(str(self.today_inspection[9])+"%")
        self.bad_quantity.change_label(str(self.today_inspection[10]))
        self.normal_quantity.change_label(str(self.today_inspection[12]))
        self.inspector_name.change_item(self.inspector_name.get_index_text(self.today_inspection[6]))
        self.workorder.change_item(self.workorder.get_index_text(self.today_inspection[13]))
        self.workorder_item.change_item(self.workorder_item.get_index_text(self.today_inspection[4], "id"))
            
        self.defect_type_count = [int(self.today_inspection[11].split(",")[0]),int(self.today_inspection[11].split(",")[1]), int(self.today_inspection[11].split(",")[2])]
        self.defect_count = self.today_inspection[10]
        self.board_count = self.today_inspection[8]
        self.defect_count_list[0] = int(self.today_inspection[11].split(",")[0])+int(self.today_inspection[11].split(",")[1])+int(self.today_inspection[11].split(",")[2])

        # self.sqldatabase.check_today_table(self.get_inspection_json())
    def stop(self, work_stop):
        if self.working:
            self.working = False
            self.quit()
            self.wait(500)
            if work_stop:
                inspection_json = {
                    "workorder_item_id": self.workorder_item.get_workorder_id(), 
                    "inspection_date": self.inspection_date.get_label(), 
                    "start_date":self.start_date.text(),
                    "end_date":self.end_date.text(),
                    "inspector_name":self.inspector_name.get_current_text(),
                    "workorder_quantity":int(self.workorder_quantity.get_label()),
                    "inspection_quatity":self.board_count,
                    "inspection_percent":int((self.board_count/int(self.workorder_quantity.get_label()))*100),
                    "bad_quantity":self.defect_count,
                    "bad_type":[{0: self.defect_type_count[0], 1: self.defect_type_count[1], 2:self.defect_type_count[2]}],
                    "normal_quantity":self.board_count-self.defect_count,       
                }
                if post_inspection(inspection_json) == 200:
                    self.sqldatabase.update_post_table(self.end_date.text(), self.today_inspection[0])

        self.sleep(0.01)
    def run(self):
        self.working=True
        # while self.working:
            # if self.defect_show_list:
                        
            #     self.defect_img_label.setPixmap(self.defect_show_list[0])
            #     self.stackedWidget.setCurrentIndex(1)
        #     ret = self.cam.MV_CC_GetOneFrameTimeout(self.data_buf, self.nPayloadSize, self.stFrameInfo, 1000)
        #     if ret == 0:
        #             # print ("get one frame: Width[%d], Height[%d], PixelType[0x%d], nFrameNum[%d]"  % (stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.enPixelType,stFrameInfo.nFrameNum))
        #             data = np.frombuffer(self.data_buf, count=int(self.stFrameInfo.nFrameLen), dtype=np.uint8)
        #             frame = image_control(data=data, stFrameInfo=self.stFrameInfo)
        #             self.board_model.img_processing(frame)
        #             self.board_model.inference()
        #             self.board_detect()
        #             # predict_list = self.board_model.process_predictions(board_pred, frame, im)

                             
        #     else:
        #          print("no data")
               
        #     self.sleep(0.01)
        dataset = LoadImages("/media/user/exFAT/test", img_size=[416,416], stride=32, auto=self.board_model.model.pt, vid_stride=1)
        for path, im, im0s, vid_cap, s in dataset:
            if self.working:
                if self.defect_show_list:
                    
                    self.defect_img_label.setPixmap(self.defect_show_list[0])
                    self.stackedWidget.setCurrentIndex(1)
                start = time.time()
                self.board_model.img_processing(im0s)
                self.board_model.inference()
                self.board_detect(path)
                end = time.time()
                print(f"detect time : {end-start}")
                    
            else:
                break
        
    def board_check(self, ob_xyxy_list, ob_xywh_list):
        ob_xyxy_list2 = []

        if len(ob_xyxy_list)>0 and len(ob_xyxy_list)%2 == 0 :

            check_list = []
            for ob_xywh in range(len(ob_xywh_list)):
                if ob_xywh not in check_list:
                    two_board_check = True
                    for check_xywh in range(ob_xywh+1, len(ob_xywh_list)):
                        if check_xywh not in check_list :
                            
                            if 0.34<ob_xywh_list[ob_xywh][0] and ob_xywh_list[ob_xywh][0]<0.45:
                                if 0.55<ob_xywh_list[check_xywh][0] and ob_xywh_list[check_xywh][0]<0.70 and abs(ob_xywh_list[ob_xywh][1]-ob_xywh_list[check_xywh][1])<=0.15:
                                    check_list.append(ob_xywh)
                                    check_list.append(check_xywh)
                                    two_board_check = True
                                    break
                                else:
                                    two_board_check=False
                            elif 0.55<ob_xywh_list[ob_xywh][0] and ob_xywh_list[ob_xywh][0]<0.70:
                                if 0.34<ob_xywh_list[check_xywh][0] and ob_xywh_list[check_xywh][0]<0.45 and abs(ob_xywh_list[ob_xywh][1]-ob_xywh_list[check_xywh][1])<=0.15:
                                    check_list.append(ob_xywh)
                                    check_list.append(check_xywh)
                                    
                                    two_board_check = True
                                    break
                                else:
                                    two_board_check = False
                            else:
                                two_board_check = False
                    if two_board_check == False:

                        ob_xyxy_list = []
                        ob_xywh_list = []
                        ob_xyxy_list2 = []
                        self.board_flag = False
                        self.defect_type_flag = [False,False,False,False,False,False]
                        self.defect_board_flag = [False,False,False,False,False,False]
                        self.defect_show_flag = False
                        break

            if len(ob_xyxy_list)>0 and self.board_flag == False:
                self.board_count += len(ob_xyxy_list)
                self.board_flag = True
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
            self.board_flag = False
            self.defect_type_flag = [False,False,False,False,False,False]
            self.defect_board_flag = [False,False,False,False,False,False]
            self.defect_show_flag = False
        return ob_xyxy_list2
    def edge_check(self, board_img, edge_count, p=None):
        edges = cv2.cvtColor(board_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(edges, 20, 30)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        linelist = []
        edge_check_list = [[board_img.shape[1]-EDGE_CHECK_SIZE[1], board_img.shape[1]-EDGE_CHECK_SIZE[0]],
                                  [EDGE_CHECK_SIZE[0], EDGE_CHECK_SIZE[1]]]
        is_even_odd = edge_count%2 
        try:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                # cv2.line(board_img, (x1, y1),
                #                   (x2, y2), (238, 255, 0), 2)
                # print(x1, x2, edge_check_list)
                # 이미지에 검출된 선을 그리기
                if x1>edge_check_list[is_even_odd][0] and x1<edge_check_list[is_even_odd][1] and abs(x1-x2)<EDGE_CHECK_SIZE[2]:
                    linelist.append([x1,y1,x2,y2])

                            # 선의 x,y 좌표 출력

                            # 이미지와 검출된 선 출력
        except:
            linelist = []
            print("error")
                    
        # print(linelist)
        linelist.sort()
        side_list = [board_img.shape[1], 0]
        board = board_img.copy()            
        if linelist :
            is_on_right_side = [len(linelist)-1, 0]
            x1_line = linelist[is_on_right_side[is_even_odd]][0]
            x2_line = linelist[is_on_right_side[is_even_odd]][2]
            # cv2.line(board_img, (linelist[is_on_right_side[is_even_odd]][0], linelist[is_on_right_side[is_even_odd]][1]),
            #                       (linelist[is_on_right_side[is_even_odd]][2], linelist[is_on_right_side[is_even_odd]][3]), (238, 255, 0), 2)
            if x1_line>edge_check_list[is_even_odd][0] and x1_line<edge_check_list[is_even_odd][1] and abs(x1_line-x2_line)<EDGE_CHECK_SIZE[2]:
                # if is_even_odd == 0 :
                                
                #     if x1_line>=x2_line and x1_line<=side_list[is_even_odd]:
                #         side_list[is_even_odd] = x1_line
                         
                #     elif x1_line<x2_line and x2_line<=side_list[is_even_odd]:
                #                 side_list[is_even_odd] = x2_line
                # else:

                #     if x1_line>=x2_line and x2_line>=0:
                #         side_list[is_even_odd] = x2_line
                #     elif x1_line<x2_line and x1_line>=0:
                #         side_list[is_even_odd] = x1_line
                # if abs(x1_line-x2_line)>=30:
                #     if is_even_odd == 0 :
                #         side_list[is_even_odd] = ((x2_line+x1_line)//2) +abs(x1_line-x2_line)//3
                #     else:
                #         side_list[is_even_odd] = ((x2_line+x1_line)//2) -abs(x1_line-x2_line)//3
                # else :
                #     side_list[is_even_odd] = (x2_line+x1_line)//2
                side_list[is_even_odd] = (x2_line+x1_line)//2
            left = side_list[1]
            right = side_list[0]
            board = board_img[0:board_img.shape[0], left:right]
        # cv2.imwrite(f"test/{p.stem}_{edge_count}.jpg", board_img)
        # cv2.imwrite(f"test/{p.stem}_{edge_count}_new.jpg", board)
        # print(p.stem, edge_count)
        # print("=====================================")
        return board
            # print("==================================")
    def board_detect(self, p=None):  
        p = Path(p)       
        for i, det in enumerate(self.board_model.pred):
            im0 = self.board_model.frame.copy()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] 
            imc = im0.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(self.board_model.im.shape[2:], det[:, :4], im0.shape).round()
                xywh_count = 0
                ob_xyxy_list = []
                ob_xywh_list = []

                
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    print(p.stem, xywh)
                    if (xywh[0]>0.45 and xywh[0]<0.57) and (xywh[2]>0.38 and xywh[2]<0.58 and xywh[3]<0.40 and xywh[3]>0.20):

                        xywh.append(xywh_count)
                        xywh_count += 1
                        crop_img=self.board_model.img_crop(xyxy, imc)


                        ob_xyxy_list.append(crop_img)
                        ob_xywh_list.append(xywh)
                
                # board_list = self.board_check(ob_xyxy_list, ob_xywh_list)
                
                result_list = []
                board_count = 0
                dtime = datetime.datetime.now()
                dtime = f"{dtime.year}{change_time_format(dtime.month)}{change_time_format(dtime.day)}{change_time_format(dtime.hour)}{change_time_format(dtime.minute)}{change_time_format(dtime.second)}"
                edge_count = 0
                self.defect_flag = False
                for board_img in ob_xyxy_list :
                    # board = self.edge_check(board_img, edge_count)
                    board_img_list = []
                    board_img_list.append(board_img[0:board_img.shape[0], 0:board_img.shape[1]//2])
                    board_img_list.append(board_img[0:board_img.shape[0], board_img.shape[1]//2:board_img.shape[1]])
                    for board in board_img_list:
                        self.defect_model.img_processing(board)
                        self.defect_model.inference()
                        for i, det in enumerate(self.defect_model.pred):
                            board_im0 = board.copy()
                            board_gn = torch.tensor(board_im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                            annotator = Annotator(board_im0, line_width=self.line_thickness, example=str(self.defect_type_list))
                            if len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_boxes(self.defect_model.im.shape[2:], det[:, :4], board_im0.shape).round()
                                # Write results
                                
                                
                                for *xyxy, conf, cls in reversed(det):
                                    xywh_defect = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / board_gn).view(-1).tolist()
                                    c = int(cls)
                                    if c != 15 and c!= 17 and c!=19 and c!=20 and c!=21:
                                        if xywh_defect[2]>0.032:
                                            if c == 1 :
                                                if xywh_defect[3]>0.044:
                                                    self.defect_flag = True
                                                    # if self.defect_board_flag[board_count] == False:
                                                    #     self.defect_count+=1#이경우에만 알람 !
                                                    #     self.defect_board_flag[board_count] = True
                                                    # if self.defect_type_flag[board_count] == False:
                                                    #             self.defect_type_count[self.defect_type[c]]+=1
                                                    # self.defect_flag = True
                                                    # self.defect_type_label = self.defect_type_list[self.defect_type[c]]
                                                    label = f'{self.defect_type_list[self.defect_type[c]]} {conf:.2f}'
                                                    label = ""
                                                    
                                                
                                                    annotator.box_label(xyxy, label, color=colors(c, True))
                                            else:
                                                # if self.defect_board_flag[board_count] == False:
                                                #         self.defect_count+=1  #이경우에만 알람 !
                                                #         self.defect_board_flag[board_count] = True
                                                # if self.defect_type_flag[board_count] == False:
                                                #             self.defect_type_count[self.defect_type[c]]+=1
                                                self.defect_flag = True
                                                # self.defect_type_label = self.defect_type_list[self.defect_type[c]]
                                                label = f'{self.defect_type_list[self.defect_type[c]]} {conf:.2f}'
                                                label = ""
                                                print(xyxy)
                                                annotator.box_label(xyxy, label, color=colors(c, True))


                            crop_im0 = annotator.result()
                            crop_im0 = cv2.resize(crop_im0, (700, 560))
                            
                            result_list.append(crop_im0)
                            # cv2.imwrite(f"test/{dtime}_{board_count}.jpg", crop_im0)
                            # crop_im0 = cv2.resize(crop_im0, (700, 560))
                            # result_list.append(crop_im0) 
                        # board_count+=1
                        # edge_count+=1
                if result_list and self.defect_flag:
                    img_list = []    
                    for i in range(len(result_list)//2):
                        img_list.append([result_list[i*2], result_list[i*2+1]])
                    save_img = cv2.vconcat([cv2.hconcat(img) for img in img_list])
                    shown_img = QImage(save_img, save_img.shape[1], save_img.shape[0], save_img.strides[0], QImage.Format_BGR888)
                    self.bad_img_label.setPixmap(QPixmap.fromImage(shown_img).scaled(1088, 928, Qt.IgnoreAspectRatio))
                    cv2.imwrite("test/"+p.stem+".jpg", save_img)
        #         if len(result_list)>0 and len(result_list)%2 == 0 and self.defect_flag:
        #             img_list = []    
        #             for i in range(len(result_list)//2):
        #                 img_list.append([result_list[i*2], result_list[i*2+1]])
        #             save_img = cv2.vconcat([cv2.hconcat(img) for img in img_list])
        #             save_img = cv2.rotate(save_img, cv2.ROTATE_180)
        #             save_hori_img = cv2.rotate(save_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #             shown_img = QImage(save_img, save_img.shape[1], save_img.shape[0], save_img.strides[0], QImage.Format_BGR888)
        #             shown_hori_img = QImage(save_hori_img, save_hori_img.shape[1], save_hori_img.shape[0], save_hori_img.strides[0], QImage.Format_BGR888)
        #             if not self.defect_show_flag :
        #                 self.defect_show_list.append(QPixmap.fromImage(shown_hori_img).scaled(save_hori_img.shape[1], 908, Qt.IgnoreAspectRatio))
        #                 self.bad_img_label.setPixmap(QPixmap.fromImage(shown_img).scaled(1088, 928, Qt.IgnoreAspectRatio))
        #                 self.defect_show_flag = True
        #             save_img = cv2.resize(save_img, (save_img.shape[0]//2, save_img.shape[1]//2))
        #             write_file = f"test/{dtime}.jpg"
        #             cv2.imwrite(write_file, self.board_model.frame)
        #             self.defect_flag = False
        #     # print(path.stem, "board count", self.board_count, "self.defect_count", sum(self.defect_count_list), "self.defect_type_count", self.defect_type_count)
        #     self.before_board_flag = self.board_flag        
        # self.inspection_quantity.change_label(f"{self.board_count}")
        # self.inspection_percent.change_label(f"{int((self.board_count/int(self.workorder_quantity.get_label()))*100)}%")
        # self.bad_quantity.change_label(f"{self.defect_count}")
        # self.bad_type.change_label(f"{self.defect_type_label}")
        # self.normal_quantity.change_label(f"{self.board_count-self.defect_count}")
        # self.sqldatabase.update_table(self.get_inspection_json(), self.today_inspection[0], self.workorder.get_current_text())
    def get_working(self):
        return self.working
    def get_inspection_json(self):
        
        inspection_json = {
                    "workorder_item_id": self.workorder_item.get_workorder_id(), 
                    "inspection_date": self.inspection_date.get_label(), 
                    "start_date":self.start_date.text(),
                    "end_date":self.end_date.text(),
                    "inspector_name":self.inspector_name.get_current_text(),
                    "workorder_quantity":int(self.workorder_quantity.get_label()),
                    "inspection_quatity":self.board_count,
                    "inspection_percent":int((self.board_count/int(self.workorder_quantity.get_label()))*100),
                    "bad_quantity":self.defect_count,
                    "bad_type":[{0: self.defect_type_count[0], 1: self.defect_type_count[1], 2:self.defect_type_count[2]}],
                    "normal_quantity":self.board_count-self.defect_count,       
                }
        return inspection_json
    def del_defect_show_list(self):
        del self.defect_show_list[0]
def change_date_format(str_time):
	if len(str(str_time))==2 :
		return str(str_time)
	else :
		return '0'+str(str_time)

def get_datetime():
    now = datetime.datetime.now()
    return f"{now.year}-{change_date_format(now.month)}-{change_date_format(now.day)} {change_date_format(now.hour)}:{change_date_format(now.minute)}:{change_date_format(now.second)}"

def get_date():
    now = datetime.datetime.now()
    return f"{now.year}-{change_date_format(now.month)}-{change_date_format(now.day)}"

class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(1920, 1020)
        self.stackedWidget = QStackedWidget()
        defect_img_widget = QWidget(self)
        defect_img_layout = QVBoxLayout(self)
        main_widget = QWidget(self)
        main_layout = QHBoxLayout(self)
        left_layout = QVBoxLayout()
        center_layout = QVBoxLayout()
        right_layout = QVBoxLayout()


        inspection_date = TitleLabel("검사 일자")
        inspection_date.change_label(get_date())

        
        inspector_name = TitleCombox("검사 담당자", 0, "name")


        self.workorder_quantity = TitleLabel("작지 수량")
        self.workorder_quantity.change_label("1000")

        inspection_quantity = TitleLabel("검사 수량")
        inspection_quantity.change_label("0")

        inspection_percent = TitleLabel("검사 진행율")
        inspection_percent.change_label("0%")

        bad_quantity = TitleLabel("불량 수량")
        bad_quantity.change_label("0")

        bad_type = TitleLabel("불량 유형")
        bad_type.change_label("")

        normal_quantity = TitleLabel("양품 수량")
        normal_quantity.change_label("0")


        # test_date_combo = QComboBox(self)
        # test_date_combo.addItems([""])

        # button = QPushButton("")
        left_layout.addWidget(inspection_date)
        left_layout.addWidget(inspector_name)
        left_layout.addWidget(self.workorder_quantity)
        left_layout.addWidget(inspection_quantity)
        left_layout.addWidget(inspection_percent)
        left_layout.addWidget(bad_quantity)
        left_layout.addWidget(bad_type)
        left_layout.addWidget(normal_quantity)
        
        self.workorder_item = TitleCombox("작지 품명", 2, "item_name", self.workorder_quantity, 1, 2 )
        self.workorder = TitleCombox("작지 번호", 1, "number",self.workorder_item, 1, 2) #작지번호에 의해서 작지 품명이 바껴야 함
        self.workorder_item.setContentsMargins(0,0,0,305)
        bad_img_label = QLabel()
        bad_img_label.setStyleSheet("border: 1px solid #374781;")
        bad_img_label.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        center_layout.setContentsMargins(0,25,0,25)
        center_layout.addWidget(bad_img_label)

        self.inspection_start_btn = QPushButton("검사 시작")
        self.inspection_start_btn.clicked.connect(self.on_inspection_start)

        self.inspection_start_time = QLabel()
        
        
        self.inspection_stop_btn = QPushButton("검사 종료")
        self.inspection_stop_btn.clicked.connect(self.on_inspection_stop)

        self.inspection_stop_time = QLabel()


        right_layout.addWidget(self.workorder)
        right_layout.addWidget(self.workorder_item)
        right_layout.addWidget(self.inspection_start_btn)
        right_layout.addWidget(self.inspection_start_time)
        right_layout.addWidget(self.inspection_stop_btn)
        right_layout.addWidget(self.inspection_stop_time)
        right_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        

        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(center_layout, 6)
        main_layout.addLayout(right_layout,3)
   

        self.inspection_start_btn.setFixedHeight(150)
        self.inspection_start_btn.setStyleSheet("color: #ffffff; font-size: 28px; background: #0d5e2d")
        self.inspection_start_time.setAlignment(Qt.AlignCenter)

        self.inspection_start_time.setFixedHeight(68)

        self.inspection_start_time.setStyleSheet("border: 1px solid #374781; color: #000000; font-size: 28px;")
        self.inspection_stop_btn.setFixedHeight(150)
        self.inspection_stop_btn.setStyleSheet("color: #ffffff; font-size: 28px; background: #9e4f00")
        self.inspection_stop_btn.setEnabled(False)
        self.inspection_stop_time.setAlignment(Qt.AlignCenter)

        self.inspection_stop_time.setFixedHeight(68)

        self.inspection_stop_time.setStyleSheet("border: 1px solid #374781; color: #000000; font-size: 28px;")
        
        right_layout.setContentsMargins(0,0, 0, 25)
        self.setStyleSheet("background: #ffffff")
        main_widget.setLayout(main_layout)
        main_layout.setContentsMargins(0,0,0,0)
        defect_img_label = QLabel()
        defect_img_label.setStyleSheet("border: 1px solid #374781;")
        defect_img_label.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        defect_show_button_hbox = QHBoxLayout()

        defect_show_button = QPushButton("확인")
        defect_show_button.setStyleSheet("color: #ffffff; font-size: 28px; background: #0d5e2d")
        defect_show_button.setFixedSize(100,60)
        defect_show_button.clicked.connect(self.on_defect_show)
        defect_show_button_hbox.addStretch(1)
        defect_show_button_hbox.addWidget(defect_show_button)
        
        defect_img_layout.addWidget(defect_img_label, 9)
        defect_img_layout.addLayout(defect_show_button_hbox, 1)
        defect_img_widget.setLayout(defect_img_layout)
        self.stackedWidget.addWidget(main_widget)
        self.stackedWidget.addWidget(defect_img_widget)

        

        layout = QVBoxLayout(self)
        layout.addWidget(self.stackedWidget)
        self.setLayout(layout)
        self.stackedWidget.setCurrentIndex(0)
        self.inspection_flag = False
        
        self.boardDefectDetect = BoardDefectDetect(self.workorder_quantity, inspection_quantity, inspection_percent, bad_quantity, 
                 bad_type, normal_quantity,bad_img_label, self.workorder, self.workorder_item, inspector_name, inspection_date, 
                 self.inspection_start_time, self.inspection_stop_time, defect_img_label, self.stackedWidget)
    def clicked_button(self, button, is_start):
        if button.text() == "네":
            if is_start:
                if not self.boardDefectDetect.get_working():
                    if self.boardDefectDetect.today_inspection[1]=='':
                        self.inspection_start_time.setText(get_datetime())
                    self.boardDefectDetect.start()
                    self.inspection_start_btn.setText("검사중..")
                    self.inspection_start_btn.setEnabled(False)
                    self.inspection_stop_btn.setEnabled(True)
            else:
                if self.boardDefectDetect.get_working():
                    self.inspection_stop_time.setText(get_datetime())
                    self.boardDefectDetect.stop(True)
                    self.inspection_start_btn.setText("검사 시작")
                    self.inspection_start_btn.setEnabled(True)
                    self.inspection_stop_btn.setEnabled(False)
        elif button.text() == "아니오":
            print("아니오 버튼이 클릭되었습니다.")

    def start_message_box(self):
        msgBox = QMessageBox()
        
        msgBox.setText("선택한 항목으로 검사를 시작하겠습니까?")
        msgBox.setStyleSheet("font-size: 28px;")
        msgBox.setWindowTitle("Start")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        msgBox.setMinimumSize(500, 500)
        
        yes_button = msgBox.button(QMessageBox.No)
        yes_button.setText("네")
        yes_button.setIcon(QIcon())
        yes_button.clicked.connect(lambda: self.clicked_button(yes_button, True))
        yes_button.setContentsMargins(0,20,0,0)
        yes_button.setStyleSheet("font-size: 24px;")
        yes_button.setFixedSize(120,50)

        no_button = msgBox.button(QMessageBox.Yes)
        no_button.setText("아니오")
        no_button.setIcon(QIcon())
        no_button.clicked.connect(lambda: self.clicked_button(no_button, True))
        no_button.setContentsMargins(100,20,0,0)
        no_button.setStyleSheet("font-size: 24px;")
        no_button.setFixedSize(120,50)
        

        
        msgBox.exec_()

    def stop_message_box(self):
        msgBox = QMessageBox()
        stop_text = "<div style='font-size: 28px;'>검사를 종료하시겠습니까?</div><div style='font-size: 20px;'><br/>* 검사 결과는 MES 서버로 전송됩니다.</div>"
        msgBox.setText(stop_text)

        msgBox.setWindowTitle("Stop")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        msgBox.setMinimumSize(500, 500)

        yes_button = msgBox.button(QMessageBox.No)
        yes_button.setText("네")
        yes_button.setIcon(QIcon())
        yes_button.clicked.connect(lambda: self.clicked_button(yes_button, False))
        yes_button.setContentsMargins(0,20,0,0)
        yes_button.setStyleSheet("font-size: 24px;")
        yes_button.setFixedSize(120,50)

        no_button = msgBox.button(QMessageBox.Yes)
        no_button.setText("아니오")
        no_button.setIcon(QIcon())
        no_button.clicked.connect(lambda: self.clicked_button(no_button, False))
        no_button.setContentsMargins(0,20,0,0)
        no_button.setStyleSheet("font-size: 24px;")
        no_button.setFixedSize(120,50)
        msgBox.exec_()

    def close_message_box(self):
        msgBox = QMessageBox()
        msgBox.setText("검사 진행중입니다.\n 프로그램 종료를 원하시면 검사 종료 버튼을 눌러주세요.")
        msgBox.setStyleSheet("font-size: 28px;")
        msgBox.setWindowTitle("Close")
        msgBox.setStandardButtons(QMessageBox.Yes)

        msgBox.setMinimumSize(500, 500)

        yes_button = msgBox.button(QMessageBox.Yes)
        yes_button.setText("네")
        yes_button.setIcon(QIcon())
        
        yes_button.setContentsMargins(0,20,0,0)
        yes_button.setStyleSheet("font-size: 24px;")
        yes_button.setFixedSize(120,50)

        
        msgBox.exec_()
    def on_inspection_start(self):
        self.start_message_box()

    def on_inspection_stop(self):
        self.stop_message_box()
        
    def closeEvent(self, event):
        if self.boardDefectDetect.get_working():
            self.close_message_box()
            event.ignore()
        else:
            self.boardDefectDetect.sqldatabase.close_db()
            event.accept()

    def on_defect_show(self):
        self.stackedWidget.setCurrentIndex(0)
        self.boardDefectDetect.del_defect_show_list()
if __name__ == '__main__':
    app = QApplication(sys.argv)

    win = Main()
    win.show()
    sys.exit(app.exec_())