
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtMultimedia import QMediaPlayer, QMediaContent, QMediaPlaylist


from qt_style import TitleLabel, TitleCombox
import sys, datetime
from mj_detector_ob_copy import BoardDefectDetect,PRODUCT_FLAG
from database import SQLDatabase
import os
SOUND_LIST = ["no_s.mp3", "no_f.mp3", "s.mp3"]
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
def change_number(num):
    if len(str(num))>3:
        return str(num)[:-3]+","+str(num)[-3:]
    return str(num)

def change_text(str_num):
    return int(str_num.replace(",", ""))
class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(1920, 1020)
        self.stackedWidget = QStackedWidget()
        defect_img_widget = QWidget(self)
        defect_img_widget.setContentsMargins(0,0,0,0)
        defect_img_layout = QHBoxLayout(self)

        # [jk] add
        camera_img_widget = QWidget(self)
        camera_img_widget.setContentsMargins(0,0,0,0)
        camera_img_layout = QHBoxLayout(self)
        # [jk] add



        main_widget = QWidget(self)
        main_layout = QHBoxLayout(self)
        left_layout = QVBoxLayout()
        center_layout = QVBoxLayout()
        right_layout = QVBoxLayout()


        self.inspection_date = TitleLabel("검사 일자")
        self.inspection_date.change_label(get_date())

        
        self.inspector_name = TitleCombox("검사 담당자", 0, "name")


        self.workorder_quantity = TitleLabel("작지 수량")
        self.workorder_quantity.change_label("100")

        self.inspection_quantity = TitleLabel("검사 수량")
        self.inspection_quantity.change_label("0")

        self.inspection_percent = TitleLabel("검사 진행율")
        self.inspection_percent.change_label("0%")

        self.bad_quantity = TitleLabel("불량 수량")
        self.bad_quantity.change_label("0")

        self.bad_type = TitleLabel("불량 유형")
        self.bad_type.change_label("")

        self.normal_quantity = TitleLabel("양품 수량")
        self.normal_quantity.change_label("0")


        # test_date_combo = QComboBox(self)
        # test_date_combo.addItems([""])

        # button = QPushButton("")
        left_layout.addWidget(self.inspection_date)
        left_layout.addWidget(self.inspector_name)
        left_layout.addWidget(self.workorder_quantity)
        left_layout.addWidget(self.inspection_quantity)
        left_layout.addWidget(self.inspection_percent)
        left_layout.addWidget(self.bad_quantity)
        left_layout.addWidget(self.bad_type)
        left_layout.addWidget(self.normal_quantity)
        
        self.workorder_item = TitleCombox("작지 품명", 2, "item_name", self.workorder_quantity, 1, 2 )
        self.workorder = TitleCombox("작지 번호", 1, "number",self.workorder_item, 1, 2) #작지번호에 의해서 작지 품명이 바껴야 함
        self.workorder_item.setContentsMargins(0,0,0,0)
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

        # [jk] add
        self.camera_view_button = QPushButton("camera view")
        self.camera_view_button.clicked.connect(self.on_camera_view)
        # [jk] add

        right_layout.addWidget(self.workorder)
        right_layout.addWidget(self.workorder_item)
        right_layout.addStretch()
        right_layout.addWidget(self.camera_view_button)
        right_layout.addWidget(self.inspection_start_btn)
        right_layout.addWidget(self.inspection_start_time)
        right_layout.addWidget(self.inspection_stop_btn)
        right_layout.addWidget(self.inspection_stop_time)
        right_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        

        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(center_layout, 6)
        main_layout.addLayout(right_layout,3)

        # [jk] add
        self.camera_view_button.setFixedHeight(70)
        self.camera_view_button.setContentsMargins(0,0,0,0)
        self.camera_view_button.setStyleSheet("color: #ffffff; font-size: 28px; background: #0d5e2d")
        # [jk] add


        self.inspection_start_btn.setFixedHeight(150)
        self.inspection_start_btn.setContentsMargins(0,0,0,0)
        self.inspection_start_btn.setStyleSheet("color: #ffffff; font-size: 28px; background: #0d5e2d")
        self.inspection_start_time.setAlignment(Qt.AlignCenter)

        self.inspection_start_time.setFixedHeight(68)

        self.inspection_start_time.setStyleSheet("border: 1px solid #374781; color: #000000; font-size: 28px;")
        self.inspection_stop_btn.setFixedHeight(150)
        self.inspection_stop_btn.setStyleSheet("color: #ffffff; font-size: 28px; background: #9e4f00")
        self.inspection_stop_btn.setEnabled(True)
        self.inspection_stop_time.setAlignment(Qt.AlignCenter)

        self.inspection_stop_time.setFixedHeight(68)

        self.inspection_stop_time.setStyleSheet("border: 1px solid #374781; color: #000000; font-size: 28px;")
        
        right_layout.setContentsMargins(0,15, 0, 25)
        self.setStyleSheet("background: #ffffff")
        main_widget.setLayout(main_layout)
        main_layout.setContentsMargins(0,0,0,0)
        # [jk] add
        camera_temp_label = QLabel()
        # [jk] add

        temp_label = QLabel()
        self.defect_img_label = QLabel()
        self.defect_img_label.setContentsMargins(0,0,0,0)
        self.defect_img_label.setStyleSheet("border: 1px solid #374781;")
        self.defect_img_label.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        defect_show_button_hbox = QHBoxLayout()
        self.defect_show_list = []

        # [jk] add
        self.camera_img_label = QLabel()
        self.camera_img_label.setContentsMargins(0,0,0,0)
        self.camera_img_label.setStyleSheet("border: 1px solid #374781;")
        self.camera_img_label.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        camera_show_button_hbox = QHBoxLayout()
        # [jk] add



        defect_show_button = QPushButton("확인")
        defect_show_button.setStyleSheet("color: #ffffff; font-size: 32px; background: #0d5e2d")
        defect_show_button.setFixedSize(160,100)
        defect_show_button.clicked.connect(self.on_defect_show)
        defect_show_button_hbox.addStretch(1)
        defect_show_button_hbox.addWidget(defect_show_button)

        # [jk] add
        camera_show_button = QPushButton("back")
        camera_show_button.setStyleSheet("color: #ffffff; font-size: 32px; background: #0d5e2d")
        camera_show_button.setFixedSize(160,100)
        camera_show_button.clicked.connect(self.on_camera_button)
        camera_show_button_hbox.addStretch(1)
        camera_show_button_hbox.addWidget(camera_show_button)
        # [jk] add

        
        defect_img_layout.setContentsMargins(0,0,0,0)
        defect_img_layout.addWidget(temp_label, 2)
        defect_img_layout.addWidget(self.defect_img_label, 9)
        defect_img_layout.addLayout(defect_show_button_hbox, 1)
        defect_img_widget.setLayout(defect_img_layout)

        # [jk] add
        camera_img_layout.setContentsMargins(0,0,0,0)
        camera_img_layout.addWidget(camera_temp_label, 2)
        camera_img_layout.addWidget(self.camera_img_label, 9)
        camera_img_layout.addLayout(camera_show_button_hbox, 1)
        camera_img_widget.setLayout(camera_img_layout)
        # [jk] add


        self.stackedWidget.addWidget(main_widget)
        self.stackedWidget.addWidget(defect_img_widget)

        # [jk] add
        self.stackedWidget.addWidget(camera_img_widget)

        layout = QVBoxLayout(self)
        layout.addWidget(self.stackedWidget)
        self.setLayout(layout)
        self.stackedWidget.setCurrentIndex(0)
        self.inspection_flag = False
        self.playlist = QMediaPlaylist()
        
        self.media_player = QMediaPlayer()
        self.inspection_stop_btn.setEnabled(False)
        self.sqldatabase = SQLDatabase()
        if PRODUCT_FLAG:
            self.sqldatabase.check_post_data()
        
        self.boardDefectDetect = BoardDefectDetect(bad_img_label, self.workorder, self.workorder_item,  self.defect_img_label, self.stackedWidget)
        
        self.today_inspection = self.sqldatabase.check_today_table(self.get_inspection_json(), self.workorder.get_current_text())
        self.ui_value_change(self.today_inspection, True)
        self.boardDefectDetect.init_value_change(self.today_inspection)
        self.boardDefectDetect.today_inspection_change(self.today_inspection)
        self.boardDefectDetect.sound_data.connect(self.detect_defect)
        #[jk]
        self.boardDefectDetect.camera_view_connect.connect(self.camera_view_event)
        self.boardDefectDetect.stacked_widget.connect(self.stacked_widget_check)
        self.boardDefectDetect.update_data.connect(self.update_data)
    def get_inspection_json(self):
        #115 : 쇼트, 116: 리드미삽, 143:냉땜

        inspection_json = {
                    "workorder_item_id": self.workorder_item.get_workorder_id(), 
                    "inspection_date": self.inspection_date.get_label(), 
                    "start_date":self.inspection_start_time.text(),
                    "end_date":self.inspection_stop_time.text(),
                    "inspector_name":self.inspector_name.get_current_text(),
                    "workorder_quantity":change_text(self.workorder_quantity.get_label()),
                    "inspection_quantity":self.boardDefectDetect.board_count,
                    "inspection_percent":int((self.boardDefectDetect.board_count/change_text(self.workorder_quantity.get_label()))*100),
                    "bad_quantity":self.boardDefectDetect.defect_count,
                    
                    "bad_type":[{"143": self.boardDefectDetect.defect_type_count[0], "116": self.boardDefectDetect.defect_type_count[1], "115":self.boardDefectDetect.defect_type_count[2]}],
                    "normal_quantity":self.boardDefectDetect.board_count-self.boardDefectDetect.defect_count,       
                }
        return inspection_json
    def ui_value_change(self, data, is_today):
        if is_today :
            self.inspection_start_time.setText(str(data[1]))
            self.inspection_date.change_label(str(data[5]))
        self.workorder_quantity.change_label(change_number(data[7]))

        self.inspection_quantity.change_label(change_number(data[8]))
        self.inspection_percent.change_label(str(data[9])+"%")
        self.bad_quantity.change_label(change_number(data[10]))
        self.normal_quantity.change_label(change_number(data[12]))
        self.inspector_name.change_item(self.inspector_name.get_index_text(data[6]))
        self.workorder.change_item(self.workorder.get_index_text(data[13]))
        self.workorder_item.change_item(self.workorder_item.get_index_text(data[4], "id"))

    def update_data(self, board_count, defect_count, defect_type_label) :
        self.inspection_quantity.change_label(change_number(board_count))
        self.inspection_percent.change_label(f"{int((board_count/change_text(self.workorder_quantity.get_label()))*100)}%")
        self.bad_quantity.change_label(change_number(defect_count))
        self.bad_type.change_label(f"{defect_type_label}")
        self.normal_quantity.change_label(change_number(board_count-defect_count))
        self.sqldatabase.update_table(self.get_inspection_json(), self.today_inspection[0], self.workorder.get_current_text())
        self.stacked_widget_check()
    # [jk] add
    def camera_view_event(self, im):
            img = QImage(im, im.shape[1], im.shape[0], im.strides[0], QImage.Format_BGR888)
            self.camera_img_label.setPixmap(QPixmap.fromImage(img).scaled(1296, 972, Qt.IgnoreAspectRatio))
    def on_camera_button(self):

            self.stackedWidget.setCurrentIndex(0)
            self.boardDefectDetect.stop_camera()
    def on_camera_view(self):

            self.stackedWidget.setCurrentIndex(2)
            self.boardDefectDetect.start()
    # [jk] add
    def stacked_widget_check(self):
        # [jk] add

        if self.boardDefectDetect.defect_show_list:
                        
                self.defect_img_label.setPixmap(self.boardDefectDetect.defect_show_list[0])
                self.stackedWidget.setCurrentIndex(1)

        # else:
        #     self.stackedWidget.setCurrentIndex(0)

    def detect_defect(self, defect):
        self.playlist.clear()

        for file_path in defect:
            
            self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(os.path.join("/opt/MVS/Samples/64/Python/GrabImage/mjai",SOUND_LIST[file_path]))))
        self.media_player.setPlaylist(self.playlist)
        self.media_player.play()

    def init_json(self):
        self.inspection_stop_time.setText("")
        self.inspection_quantity.change_label("0")
        self.inspection_percent.change_label("0%")
        self.bad_quantity.change_label("0")
        self.bad_type.change_label("")
        self.normal_quantity.change_label("0")
        
        self.boardDefectDetect.defect_flag = False
        self.boardDefectDetect.defect_type_flag = [False,False,False,False,False,False]
        self.boardDefectDetect.defect_board_flag = [False,False,False,False,False,False]
        self.boardDefectDetect.defect_count_list = [0,0,0,0,0,0]
        self.boardDefectDetect.defect_alarm = False
        self.boardDefectDetect.board_count = 0 
        self.boardDefectDetect.defect_count = 0 
        self.boardDefectDetect.defect_type_label = ""
        self.boardDefectDetect.board_flag = False
        self.boardDefectDetect.defect_type_count = [0,0,0]
        self.boardDefectDetect.defect_show_flag = False
    def create_table(self):
        self.sqldatabase.insert_table(self.get_inspection_json(), self.workorder.get_current_text())
        result = self.sqldatabase.select_today_table()
        self.today_inspection = result[len(result)-1]
        self.boardDefectDetect.today_inspection_change(self.today_inspection)
    def clicked_button(self, button, is_start, commit = None):

        if button.text() == "네":
            commit_data = self.boardDefectDetect.sqldatabase.select_commit_table(self.workorder_item.get_workorder_id(),self.workorder.get_current_text())
            print(commit_data)
            
            if commit_data :
                self.start_start_message_box(commit_data[0])
            else :
                if not self.boardDefectDetect.get_working():
                    if self.boardDefectDetect.is_post == True:
                        self.init_json()
                        self.inspection_start_time.setText(get_datetime())
                        self.create_table()
                    else:

                        if self.boardDefectDetect.today_inspection[1]=='':
                            self.inspection_start_time.setText(get_datetime())
                    self.boardDefectDetect.start()
                    self.inspection_start_btn.setText("검사중..")
                    self.camera_view_button.setEnabled(False)
                    self.inspection_start_btn.setEnabled(False)
                    self.inspection_stop_btn.setEnabled(True)
            #self.start_start_message_box()
        elif button.text() == "확인":
            if not self.boardDefectDetect.get_working():
                self.boardDefectDetect.sqldatabase.update_commit_table(commit[0])
                self.ui_value_change(commit, False)
                self.boardDefectDetect.init_value_change(commit)
                if self.boardDefectDetect.today_inspection[1]=='':
                    self.inspection_start_time.setText(get_datetime())
                self.boardDefectDetect.start()
                self.inspection_start_btn.setText("검사중..")
                self.camera_view_button.setEnabled(False)
                self.inspection_start_btn.setEnabled(False)
                self.inspection_stop_btn.setEnabled(True)
        elif button.text() == "새로 시작":
            if not self.boardDefectDetect.get_working():
                self.boardDefectDetect.sqldatabase.update_commit_table(commit[0])
                
                if self.boardDefectDetect.today_inspection[1]=='':
                    self.inspection_start_time.setText(get_datetime())
                self.boardDefectDetect.start()
                self.inspection_start_btn.setText("검사중..")
                self.camera_view_button.setEnabled(False)
                self.inspection_start_btn.setEnabled(False)
                self.inspection_stop_btn.setEnabled(True)
        elif button.text() == "종료":
            self.stop_stop_message_box()
            
        elif button.text() == "일시정지":
            if self.boardDefectDetect.get_working():

                self.boardDefectDetect.stop(self.get_inspection_json(),False)
                self.inspection_start_btn.setText("검사 시작")
                self.camera_view_button.setEnabled(True)
                self.inspection_start_btn.setEnabled(True)
                self.inspection_stop_btn.setEnabled(False)
        elif button.text() == "완료":
            if self.boardDefectDetect.get_working():
                self.inspection_stop_time.setText(get_datetime())
                self.boardDefectDetect.stop(self.get_inspection_json(),True, self.inspection_stop_time.text(),True)
                self.inspection_start_btn.setText("검사 시작")
                self.camera_view_button.setEnabled(True)
                self.inspection_start_btn.setEnabled(True)
                self.inspection_stop_btn.setEnabled(False)
        elif button.text() == "저장":
            if self.boardDefectDetect.get_working():
                self.inspection_stop_time.setText(get_datetime())
                self.boardDefectDetect.stop(self.get_inspection_json(),True, self.inspection_stop_time.text(),False)
                self.inspection_start_btn.setText("검사 시작")
                self.camera_view_button.setEnabled(True)
                self.inspection_start_btn.setEnabled(True)
                self.inspection_stop_btn.setEnabled(False)
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
    def start_start_message_box(self, commit_data):
        msgBox = QMessageBox()
        
        msgBox.setText(f'{commit_data[5]} 날짜로 해당 작지 품목의 이전 검사가 남아있습니다. 이어서 진행하시겠습니까?')
        msgBox.setStyleSheet("font-size: 28px;")
        msgBox.setWindowTitle("Start")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        msgBox.setMinimumSize(500, 500)
        
        yes_button = msgBox.button(QMessageBox.No)
        yes_button.setText("확인")
        yes_button.setIcon(QIcon())
        yes_button.clicked.connect(lambda: self.clicked_button(yes_button, True, commit_data))
        yes_button.setContentsMargins(0,20,0,0)
        yes_button.setStyleSheet("font-size: 24px;")
        yes_button.setFixedSize(120,50)

        no_button = msgBox.button(QMessageBox.Yes)
        no_button.setText("새로 시작")
        no_button.setIcon(QIcon())
        no_button.clicked.connect(lambda: self.clicked_button(no_button, True, commit_data))
        no_button.setContentsMargins(100,20,0,0)
        no_button.setStyleSheet("font-size: 24px;")
        no_button.setFixedSize(120,50)
        msgBox.exec_()
    def stop_stop_message_box(self):
        msgBox = QMessageBox()
        msgBox.setText("해당 작업품목의 검사 진행을 완료하였으면 '완료'를, 검사가 남아있는 경우 '저장'을 눌러주세요. ")
        msgBox.setStyleSheet("font-size: 28px;")
        # stop_text = "<div style='font-size: 28px;'>해당 보드의 검사를 끝내길 원하시면 '종료'를, 검사가 남아있는 경우 '저장'을 눌러주세요. </div>"
        # msgBox.setText(stop_text)

        msgBox.setWindowTitle("Stop")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)

        msgBox.setMinimumSize(500, 500)

        yes_button = msgBox.button(QMessageBox.Cancel)
        yes_button.setText("완료")
        yes_button.setIcon(QIcon())
        yes_button.clicked.connect(lambda: self.clicked_button(yes_button, False))
        yes_button.setContentsMargins(0,20,0,0)
        yes_button.setStyleSheet("font-size: 24px;")
        yes_button.setFixedSize(120,50)

        no_button = msgBox.button(QMessageBox.No)
        no_button.setText("저장")
        no_button.setIcon(QIcon())
        no_button.clicked.connect(lambda: self.clicked_button(no_button, False))
        no_button.setContentsMargins(0,20,0,0)
        no_button.setStyleSheet("font-size: 24px;")
        no_button.setFixedSize(120,50)

        cancel_button = msgBox.button(QMessageBox.Yes)
        cancel_button.setText("취소")
        cancel_button.setIcon(QIcon())
        cancel_button.clicked.connect(lambda: self.clicked_button(cancel_button, False))
        cancel_button.setContentsMargins(0,20,0,0)
        cancel_button.setStyleSheet("font-size: 24px;")
        cancel_button.setFixedSize(120,50)
        msgBox.exec_()
    def stop_message_box(self):
        msgBox = QMessageBox()
        stop_text = "<div style='font-size: 28px;'>검사를 종료하시겠습니까?</div><div style='font-size: 20px;'><br/>* 검사 종료 후 검사 결과는 MES 서버로 전송됩니다.</div>"
        msgBox.setText(stop_text)

        msgBox.setWindowTitle("Stop")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)

        msgBox.setMinimumSize(500, 500)

        yes_button = msgBox.button(QMessageBox.Cancel)
        yes_button.setText("종료")
        yes_button.setIcon(QIcon())
        yes_button.clicked.connect(lambda: self.clicked_button(yes_button, False))
        yes_button.setContentsMargins(0,20,0,0)
        yes_button.setStyleSheet("font-size: 24px;")
        yes_button.setFixedSize(120,50)

        no_button = msgBox.button(QMessageBox.No)
        no_button.setText("일시정지")
        no_button.setIcon(QIcon())
        no_button.clicked.connect(lambda: self.clicked_button(no_button, False))
        no_button.setContentsMargins(0,20,0,0)
        no_button.setStyleSheet("font-size: 24px;")
        no_button.setFixedSize(120,50)

        cancel_button = msgBox.button(QMessageBox.Yes)
        cancel_button.setText("취소")
        cancel_button.setIcon(QIcon())
        cancel_button.clicked.connect(lambda: self.clicked_button(cancel_button, False))
        cancel_button.setContentsMargins(0,20,0,0)
        cancel_button.setStyleSheet("font-size: 24px;")
        cancel_button.setFixedSize(120,50)
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
        # self.stackedWidget.setCurrentIndex(0)
        self.boardDefectDetect.del_defect_show_list()
        if self.boardDefectDetect.defect_show_list:

            self.defect_img_label.setPixmap(self.boardDefectDetect.defect_show_list[0])
        else:
            self.stackedWidget.setCurrentIndex(0)
    def keyPressEvent(self,event):
        if event.key() == Qt.Key_B:
            self.boardDefectDetect.del_defect_show_list()
            if self.boardDefectDetect.defect_show_list:

                self.defect_img_label.setPixmap(self.boardDefectDetect.defect_show_list[0])
            else:
                self.stackedWidget.setCurrentIndex(0)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    win = Main()
    win.show()
    sys.exit(app.exec_())
