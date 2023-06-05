import requests, json
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
def change_number(num):
    if len(str(num))>4:
        return str(num)[:-3]+","+str(num)[-3:]
    return str(num)

def get_inspector():
    try:
        url = "https://mj.d-triple.com/api/mes/v1/external/inspector"
        headers = {"Content-Type": "application/json"}
        response = requests.get(url, headers=headers)
    
        return json.loads(response.text.encode().decode('unicode-escape'))["objects"]

    except:
        return []

def get_workorder():
    try:
        url = "https://mj.d-triple.com/api/mes/v1/external/workorder"
        headers = {"Content-Type": "application/json"}
        response = requests.get(url, headers=headers)
    
        return json.loads(response.text.encode().decode('unicode-escape'))["objects"]
    except:
        return []

def get_workorder_list(id):
    try:
        url = f"https://mj.d-triple.com/api/mes/v1/external/workorder/{id}"
        headers = {"Content-Type": "application/json"}
        response = requests.get(url, headers=headers)

        return json.loads(response.text.encode().decode('unicode-escape'))["objects"]
    except:
        return []
# def post_inspection():
#     try:
#         url = f"https://mj.d-triple.com/api/mes/v1/external/inspection"
#         headers = {"Content-Type": "application/json"}
#         inspection_json = {
#             "workorder_imte_id":, 
#             "inspection_date":, 
#             "start_date":,
#             "end_date":,
#             "inspector_name":,
#             "workorder_quantity":,
#             "inspection_quatity":,
#             "inspection_percent":,
#             "bad_quantity":,
#             "bad_type":,
#             "normal_quantity":,

            
#             }
#         response = requests.post(url, headers=headers, json={"workorder_imte_id":,})
        
#     except:
#         print("error")
def request_list(num, id=None):
    if num == 0 :
        return get_inspector()
    elif num == 1 :
        return get_workorder()
    else:
        return get_workorder_list(id)
    
class TitleLabel(QWidget):
    def __init__(self, title, style=True):
        super().__init__()

        self.layout = QHBoxLayout()
        self.title = QLabel(title)
        self.label = QLabel()

        if style:
            self.label.setAlignment(Qt.AlignCenter)
            self.title.setAlignment(Qt.AlignCenter)
            self.label.setFixedHeight(68)
            self.title.setFixedHeight(68)
            self.title.setStyleSheet("border: 1px solid #374781; color: #ffffff; font-size: 32px; background: #2f68d8;")
            self.label.setStyleSheet("border: 1px solid #374781; color: #000000; font-size: 32px; " )
        
        else:
            self.label.setFixedHeight(10)
            self.title.setFixedHeight(10)
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

    def change_label(self, text):
        self.label.setText(text)
    def get_label(self):
        return self.label.text()
class TitleCombox(QWidget):
    def __init__(self, title, num, item, other_widget=None, title_ratio=1, combobox_ratio=1):
        super().__init__()

        self.combobox_content = ""
        self.num = num
        self.item = item
        self.other_widget = other_widget
        self.item_json_list = []
        self.item_list = []
        self.layout = QHBoxLayout()
        self.title = QLabel(title)
        self.combobox = QComboBox()

        # self.combobox.setEditable(True)
        # self.combobox.lineEdit().setAlignment(Qt.AlignCenter)
        # self.combobox.lineEdit().setReadOnly(True)
        self.combobox.setFixedHeight(68)
        
        self.combobox.setStyleSheet("border: 1px solid #374781; color: #000000; font-size: 28px;")
        self.combobox.currentIndexChanged.connect(self.on_workorder_combobox_changed)

        if self.num != 2:
            self.combobox_item_add()

        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFixedHeight(68)
        self.title.setStyleSheet("border: 1px solid #374781; color: #ffffff; font-size: 28px; background: #2f68d8;")
        
        self.layout.addWidget(self.title, title_ratio)
        self.layout.addWidget(self.combobox, combobox_ratio)
        self.setLayout(self.layout)

    def combobox_item_add(self, id=None):
        self.item_json_list = request_list(self.num, id)
        self.combobox.clear()
        self.item_list = []
        for i in self.item_json_list:
            self.item_list.append(i[self.item])
        self.combobox.addItems(self.item_list)


    def workorder_item_combobox_change(self, id):
        workorder_list_list = get_workorder_list(id)
        workorder_item_list = []
        for i in workorder_list_list:
             workorder_item_list.append(i[""])
        
    def add_item(self, item):
        self.combobox.addItems(item)

    def get_combox(self):
        self.combobox.currentText()

    def get_index_text(self, text, item=None):
        text_item = self.item
        if item != None:
            text_item = item
        for i in range(len(self.item_json_list)):
            if self.item_json_list[i][text_item] == text:
                break
        return i
    
    def on_workorder_combobox_changed(self):
        if self.num == 1:
            workorder_id = None
            for i in self.item_json_list:
                
                if i["number"] == self.get_current_text() :
                    workorder_id = i["id"]
            self.other_widget.combobox_item_add(workorder_id)
        elif self.num == 2:
            for i in self.item_json_list:
                if i["item_name"] == self.get_current_text():
                    
                    self.other_widget.change_label(change_number(i["required_quantity"]))
    def get_workorder_id(self):
        workorder_item_id = None
        if self.num == 2:
            
            for i in self.item_json_list:
                if i["item_name"] == self.get_current_text() :
                    workorder_item_id = i["id"] 
        return workorder_item_id
    def get_current_text(self):
        return self.combobox.currentText()

    def change_item(self, index):

        self.combobox.setCurrentIndex(index)