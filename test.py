from PySide2.QtCore import QUrl
from PySide2.QtMultimedia import QMediaPlayer, QMediaContent
from PySide2.QtWidgets import QApplication, QPushButton, QWidget
import os, sys
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        button = QPushButton("Play Sound", self)
        button.clicked.connect(self.play_sound)
        self.media_player = QMediaPlayer()
        
        self.show()

    def play_sound(self):
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.join("/opt/MVS/Samples/64/Python/GrabImage/yolov5_","sound.mp3"))))
        self.media_player.play()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())