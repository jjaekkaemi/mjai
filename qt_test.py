
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
import sys
class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(1920, 1080)
        main_layout = QHBoxLayout(self)
        
        self.setLayout(main_layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    win = Main()
    win.show()
    sys.exit(app.exec_())