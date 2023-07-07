
from PySide2.QtCore import *
from PySide2.QtGui import *


from mj_detect import EdgeTpuModel, getSavePathDir

import os, sys, torch, time, datetime
from pathlib import Path
import numpy as np

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (cv2,scale_boxes,xyxy2xywh)
from utils.plots import Annotator, colors

import cv2
from database import SQLDatabase

PRODUCT_FLAG = True
MODEL_PATH = ""
if PRODUCT_FLAG:
    sys.path.append("../../MvImport")
    from MvCameraControl_class import *
    import detect_alarm
    import mes_requests as mes
    MODEL_PATH = "/opt/MVS/Samples/64/Python/GrabImage/mjai/"
TEST_IMG_PATH = "/media/pi/Samsung USB/detect/120S"

# [jk] add
stacked_widget_page = 0

IPO_POS_CHECK = [0.34, 0.51, 0.44, 0.68, 0.23, 0.25]
IPG_NPE_CHECK = [0.20, 0.79]
IPO_LIST_LENTH = 2
IPGNPE_LIST_LENTH = 5
RESULT_SIZE = [992, 1088, 928]
CROP_IMG_SIZE = (700,560)
IPGNPE_CROP_IMG_SIZE = (352, 292)

IPO_WORKORDER_CHECK = {
    "230M": [[0.16, 0.24], [0.72, 0.82]],
    "TYPE": [[0.54, 0.64], [0.34, 0.43]],
    "230S-": [[0.60, 0.70], [0.30, 0.40]],
    "120S": [[0.72, 0.91], [0.08,0.25]]
    }
def change_time_format(str_time):
	if len(str(str_time))==2 :
		return str(str_time)
	else :
		return '0'+str(str_time)

        
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
    sound_data = Signal(list)
    # [jk] add
    camera_view_connect = Signal(np.ndarray)
    stacked_widget = Signal()
    ui_change = Signal(dict, bool)
    update_data = Signal(int, int, str)
    def __init__(self, bad_img_label, workorder, workorder_item,defect_img_label, stackedWidget):
        super().__init__()
        if PRODUCT_FLAG:
            cam, data_buf, stFrameInfo, nPayloadSize  = initMvCamera()
            self.cam = cam
            self.data_buf = data_buf
            self.stFrameInfo = stFrameInfo
            self.nPayloadSize = nPayloadSize

        board_weights = f'{MODEL_PATH}best.pt'
        board_name = f'{MODEL_PATH}data/board.yaml'
        defect_weights = f'{MODEL_PATH}230706_ipo_defect-int8_edgetpu.tflite'
        defect_name = f'{MODEL_PATH}data/defect.yaml'
        ipgnpe_board_weights = f'{MODEL_PATH}ipgnpe_oneboard.pt'
        ipgnpe_defect_weights = f'{MODEL_PATH}230706_ipgnpe_defect.pt'
        type_board_weights = f'{MODEL_PATH}type_oneboard.pt'
        board_conf_thres = 0.7

        ipo_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15, 18,20,21]

        ipgnpe_classes = [0,1,4,5,7,8]
        ipgnpe_iou_thres = 0.2
        self.board_model = EdgeTpuModel(board_weights, board_name, conf_thres=board_conf_thres)
        self.defect_model = EdgeTpuModel(defect_weights, defect_name, classes=ipo_classes, iou_thres =ipgnpe_iou_thres)
        self.ipgnpe_board_model = EdgeTpuModel(ipgnpe_board_weights, defect_name, conf_thres=0.5)
        self.ipgnpe_defect_model = EdgeTpuModel(ipgnpe_defect_weights, defect_name, classes=ipgnpe_classes, iou_thres =ipgnpe_iou_thres)
        self.type_board_model = EdgeTpuModel(type_board_weights, board_name, conf_thres=board_conf_thres)
        self.working = False
        
        self.line_thickness = 6


        self.defect_count_list = [0,0,0,0,0,0]
        self.defect_alarm = False
        self.total_board_count = 0 
        self.defect_count = 0 
        self.defect_type_label = ""


        self.defect_type = [0, 0, 0, 0, 0, 0, 0,1, 1,0, 2,0,0, 1, 3, 0, 0, 3, 1, 3, 1, 2, 1]
        self.ipgnpe_defect_type = [0,2,3,0,0,1,3,0,1,3,3]
        self.defect_type_count = [0,0,0]

        self.defect_type_list = ["미납","리드미삽", "쇼트", ""]
        self.defect_show_list = []


        self.file_write = None
        self.defect_img_label = defect_img_label
        self.stackedWidget = stackedWidget


        self.bad_img_label = bad_img_label
        self.workorder_item = workorder_item
        
        self.sqldatabase = SQLDatabase()
        self.is_post = False
        
        
        self.board_check_flag = 0

        self.select_board_model = self.board_model
        self.select_defect_model = self.defect_model
        self.select_defect_type = self.defect_type
        #self.today_inspection = self.sqldatabase.check_today_table(self.get_inspection_json(), self.workorder.get_current_text())
        # self.ui_change.emit(self.today_inspection, True)
        # self.ui_value_change(self.today_inspection, True)
        #self.init_value_change(self.today_inspection)

        #self.sqldatabase.insert_example_table()

        # [jk] add
        self.camera_working = False
        self.ori_path_dir, self.defect_path_dir = getSavePathDir()

    def today_inspection_change(self, today_inspection):
         self.today_inspection = today_inspection
    
    def stop_camera(self):
        if self.camera_working :
            self.camera_working = False
            self.quit()
            self.wait(500)
    def stop(self, inspection_json, work_stop, end_date=None,is_commit=False):
        
        if self.working:
            self.working = False
            self.quit()
            self.wait(500)
            self.file_write.close()
            is_db_update = True
            if work_stop:
                if PRODUCT_FLAG:
                    if mes.post_inspection(inspection_json) != 200:
                        is_db_update = False
                if is_db_update :
                    self.is_post = True
                    if is_commit:
                        self.sqldatabase.update_commit_table(self.today_inspection[0])
                    self.sqldatabase.update_post_table(end_date, self.today_inspection[0])
            self.sleep(0.01)
    def run(self):
        # [jk] add
        if self.stackedWidget.currentIndex() == 2:
                self.camera_working=True
                if PRODUCT_FLAG :
                    while self.camera_working:
                        ret = self.cam.MV_CC_GetOneFrameTimeout(self.data_buf, self.nPayloadSize, self.stFrameInfo, 1000)
                        if ret == 0:
                                data = np.frombuffer(self.data_buf, count=int(self.stFrameInfo.nFrameLen), dtype=np.uint8)
                                frame = image_control(data=data, stFrameInfo=self.stFrameInfo)
                                self.camera_view_connect.emit(frame)
                        else:
                                print("no data")
                        self.sleep(0.01)
                else:
                    dataset = LoadImages(TEST_IMG_PATH, img_size=[416,416], stride=32, auto=self.select_board_model.model.pt, vid_stride=1)
                    for path, im, im0s, vid_cap, s in dataset:
                        if self.camera_working:
                            self.camera_view_connect.emit(im0s)
                                
                        else:
                            break
        else:
            if self.workorder_item.get_current_text().find("IPG") != -1:
                self.select_board_model = self.ipgnpe_board_model
                self.select_defect_model = self.ipgnpe_defect_model
                self.select_defect_type = self.ipgnpe_defect_type
            elif self.workorder_item.get_current_text().find("TYPE") != -1:
                self.select_board_model = self.type_board_model
            elif self.workorder_item.get_current_text().find("120S") != -1:
                self.select_defect_model.add_classes(16,16)
                self.select_defect_model.add_classes(22,22)
                

            self.file_write = open("log.txt", "a")
            self.is_post = False
            self.working=True
            self.defect_show_list = [] 

            if PRODUCT_FLAG:  
                while self.working:
                    
                    ret = self.cam.MV_CC_GetOneFrameTimeout(self.data_buf, self.nPayloadSize, self.stFrameInfo, 1000)
                    if ret == 0:
                            # print ("get one frame: Width[%d], Height[%d], PixelType[0x%d], nFrameNum[%d]"  % (stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.enPixelType,stFrameInfo.nFrameNum))
                            data = np.frombuffer(self.data_buf, count=int(self.stFrameInfo.nFrameLen), dtype=np.uint8)
                            frame = image_control(data=data, stFrameInfo=self.stFrameInfo)
                            self.select_board_model.img_processing(frame)
                            self.select_board_model.inference()
                            if self.workorder_item.get_current_text().find("IPG") == -1:
                                self.board_detect(True)
                            else:    
                                self.board_detect(False)
                    else:
                        print("no data")
                    
                    self.sleep(0.01)
            else:
                dataset = LoadImages(TEST_IMG_PATH, img_size=[416,416], stride=32, auto=self.select_board_model.model.pt, vid_stride=1)
                for path, im, im0s, vid_cap, s in dataset:
                    if self.working:
                        start = time.time()
                        self.select_board_model.img_processing(im0s)
                        self.select_board_model.inference()
                        if self.workorder_item.get_current_text().find("IPG") == -1:
                            self.board_detect(True, path)
                        else:    
                            self.board_detect(False, path)
                        end = time.time()
                        print(f"detect time : {end-start}")
                            
                    else:
                        break
    def board_check(self, IPO, list_lenth, ob_xyxy_list, ob_xywh_list):
        if IPO :
            if len(ob_xyxy_list)==0 or len(ob_xyxy_list)%2 != 0 :
                return []
        else :
            if len(ob_xyxy_list)!=25:
                return []
        sorted_data = sorted(ob_xywh_list, key = lambda x: x[1])
        final_sorted_data = []
        new_xyxy_list = []
        for i in range(0, len(sorted_data), list_lenth):
            group = sorted_data[i:i+list_lenth]
            group_sorted = sorted(group, key = lambda x: x[0] * x[1])
            final_sorted_data.extend(group_sorted)
        for fi in range(len(final_sorted_data)):
            if IPO :
                if final_sorted_data[fi][0]<=IPO_POS_CHECK[fi%2] or final_sorted_data[fi][0]>=IPO_POS_CHECK[(fi%2)+2]:
                    return []
            new_xyxy_list.append(ob_xyxy_list[final_sorted_data[fi][4]])
        return new_xyxy_list
       
    def ipo_work_check(self, c, xywh_defect, board_count):
        if xywh_defect[2]>0.032:
            if c == 1 and xywh_defect[3]>0.044:
                return True
            else:
                for ipo in IPO_WORKORDER_CHECK:
                    if self.workorder_item.get_current_text().find(ipo) != -1:
                        if c == 18 or c == 2:
                            if xywh_defect[0]<=IPO_WORKORDER_CHECK[ipo][board_count%2][0] or xywh_defect[0]>=IPO_WORKORDER_CHECK[ipo][board_count%2][1]:
                                return True
                        else:
                            return True
                    else:
                        return True
        return False
    def defect_detect(self, result_list, list_lenth, defect_list, dtime):
        img_list = [result_list[i:i+list_lenth] for i in range(0, len(result_list), list_lenth)]
        save_img = cv2.vconcat([cv2.hconcat(img) for img in img_list])
        save_img = cv2.rotate(save_img, cv2.ROTATE_180)
   
        shown_img = QImage(save_img, save_img.shape[1], save_img.shape[0], save_img.strides[0], QImage.Format_BGR888)
        self.defect_show_list.append(QPixmap.fromImage(shown_img).scaled(save_img.shape[1], RESULT_SIZE[0], Qt.IgnoreAspectRatio))
        self.bad_img_label.setPixmap(QPixmap.fromImage(shown_img).scaled(RESULT_SIZE[1], RESULT_SIZE[2], Qt.IgnoreAspectRatio))
        if PRODUCT_FLAG :
            detect_alarm.buzzer_on()
            self.sound_data.emit(defect_list)

            cv2.imwrite(os.path.join(self.defect_path_dir, dtime+".jpg"), save_img)
    def board_detect(self, IPO = True, p=None):     
        if not PRODUCT_FLAG:
            p = Path(p)
        for i, det in enumerate(self.select_board_model.pred):
            im0 = self.select_board_model.frame.copy()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] 
            imc = im0.copy()
            
            center_check = False
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(self.select_board_model.im.shape[2:], det[:, :4], im0.shape).round()
                xywh_count = 0
                ob_xyxy_list = []
                ob_xywh_list = []

                
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    center = False
                    if IPO:
                        if (xywh[2]>IPO_POS_CHECK[4] and xywh[3]>IPO_POS_CHECK[5]) and (xywh[0]>IPO_POS_CHECK[0] and xywh[0]<IPO_POS_CHECK[3]):
                            center = True
                            
                    else:
                        c = int(cls)
                        if c == 0 :
                            if xywh[0]>IPG_NPE_CHECK[0] and xywh[0]<IPG_NPE_CHECK[1]:
                                center = True
                    if center:
                        xywh.append(xywh_count)
                        xywh_count += 1
                        crop_img=self.select_board_model.img_crop(xyxy, imc) ###


                        ob_xyxy_list.append(crop_img)
                        ob_xywh_list.append(xywh)
                board_list = self.board_check(IPO, IPO_LIST_LENTH if IPO else IPGNPE_LIST_LENTH, ob_xyxy_list, ob_xywh_list) 
                if board_list :
                    center_check = True
                    if self.board_check_flag == 0 : # [comment] 센터에 온 경우, 첫 번째에만 검출되도록
                        self.board_check_flag = 1
                        self.total_board_count += len(board_list)
                    elif self.board_check_flag == 1: # [comment] 센터에 온 경우, 두 번째부터는 검출 안되도록
                        self.board_check_flag = 2
                else: # [comment] 센터 아닌 경우 변수 리셋
                    xywh_count = 0
                    ob_xyxy_list = []
                    ob_xywh_list = []
                    center_check = False
                    self.board_check_flag = 0

                
                # cv2.imwrite(f"/home/pi/test/{p.stem}_{board_count}.jpg", board)
                # board_count+=1
                if center_check and self.board_check_flag == 1: # [comment] 첫 센터에 온 경우 불량 검출
                    defect_check_count = False # [comment] 전체 기판에서 불량이 하나라도 나온 경우, 검출 알림을 위해 체크
                    board_count = 0 
                    result_list = []
                    defect_list = []
                    dtime = datetime.datetime.now()
                    dtime = f"{dtime.year}{change_time_format(dtime.month)}{change_time_format(dtime.day)}{change_time_format(dtime.hour)}{change_time_format(dtime.minute)}{change_time_format(dtime.second)}"
                    
                    for board in board_list :
                        defect_check = False  # [comment] 보드 하나에서 불량이 하나라도 나온 경우 불량 수 count
                        self.select_defect_model.img_processing(board)
                        self.select_defect_model.inference()
                        for i, det in enumerate(self.select_defect_model.pred):
                            board_im0 = board.copy()
                            board_gn = torch.tensor(board_im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                            annotator = Annotator(board_im0, line_width=self.line_thickness, example=str(self.defect_type_list))
                            if len(det):
                                self.file_write.write(f"\r\n\r\n======== {dtime} ========\r\n")
                                
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_boxes(self.select_defect_model.im.shape[2:], det[:, :4], board_im0.shape).round()
                                # Write results

                                for *xyxy, conf, cls in reversed(det):
                                    xywh_defect = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / board_gn).view(-1).tolist()
                                    c = int(cls)
                                    

                                    if self.ipo_work_check(c, xywh_defect, board_count) if IPO else True:
                                        self.file_write.write(f"{self.select_defect_model.model.names[c]}  {conf} {xywh_defect}\r\n")
                                        defect_check = True
                                        defect_check_count = True
                                        self.defect_type_count[self.select_defect_type[c]]+=1


                                        self.defect_type_label = self.defect_type_list[self.select_defect_type[c]]
                                        defect_list.append(self.select_defect_type[c])
                                        label = ""
                                        annotator.box_label(xyxy, label, color=colors(c, True), classes=self.select_defect_type[c])

                            crop_im0 = annotator.result()
                            
                            crop_im0 = cv2.resize(crop_im0, CROP_IMG_SIZE if IPO else IPGNPE_CROP_IMG_SIZE)
                            result_list.append(crop_im0) 


                        if defect_check :
                            self.defect_count +=1
                        board_count+=1
                    if defect_check_count :
                        self.defect_detect(result_list, IPO_LIST_LENTH if IPO else IPGNPE_LIST_LENTH, defect_list, dtime)
                cv2.imwrite(os.path.join(self.ori_path_dir, dtime+".jpg"), self.select_board_model.frame)
            # print(path.stem, "board count", self.total_board_count, "self.defect_count", sum(self.defect_count_list), "self.defect_type_count", self.defect_type_count)
   
        self.update_data.emit(self.total_board_count, self.defect_count, self.defect_type_label)
    
    def get_working(self):
        return self.working
    
    def del_defect_show_list(self):
        if len(self.defect_show_list)>0:
            del self.defect_show_list[0]


    def init_value_change(self, data):
        self.defect_type_count = [int(data[11].split(",")[0]),int(data[11].split(",")[1]), int(data[11].split(",")[2])]
        self.defect_count = data[10]
        self.total_board_count = data[8]
        self.defect_count_list[0] = int(data[11].split(",")[0])+int(data[11].split(",")[1])+int(data[11].split(",")[2])
