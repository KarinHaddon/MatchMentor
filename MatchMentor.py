"""Contains code for running the MatchMentor application."""

import sys

import cv2
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class VideoPlayer(QMainWindow):

    def __init__(self):
        super().__init__()

        self.title = "MatchMentor"
        self.left = 100
        self.top = 100
        self.width = 720
        self.height = 480
        self.initUI()

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

    @pyqtSlot()
    def play_video(self):
        if self.cap is not None and not self.timer.isActive():
            self.timer.start(1000 / 30)  # Update the frame every 1/30th of a second

    @pyqtSlot()
    def pause_video(self):
        if self.timer.isActive():
            self.timer.stop()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create widget
        widget = QWidget(self)
        layout = QVBoxLayout()

        # Video frame label
        self.video_frame_label = QLabel(self)
        layout.addWidget(self.video_frame_label)

        # Status bar
        self.status_bar = QLabel("Frame: 0 / Time: 0.00s", self)
        layout.addWidget(self.status_bar)

        # Load video button
        load_video_btn = QPushButton("Load Video", self)
        load_video_btn.clicked.connect(self.load_video)
        layout.addWidget(load_video_btn)

        # Frame Info
        frame_info_layout = QVBoxLayout()

        self.frame_info_title = QLabel("Frame Info")
        frame_info_layout.addWidget(self.frame_info_title)

        self.frame_dropdown = QComboBox(self)
        frame_info_layout.addWidget(self.frame_dropdown)

        next_frame_btn = QPushButton("Next Frame", self)
        next_frame_btn.clicked.connect(self.next_frame)
        frame_info_layout.addWidget(next_frame_btn)

        prev_frame_btn = QPushButton("Previous Frame", self)
        prev_frame_btn.clicked.connect(self.previous_frame)
        frame_info_layout.addWidget(prev_frame_btn)

        # Put Frame Info to the right of video
        main_layout = QHBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(frame_info_layout)

        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        # Member variables
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0

    @pyqtSlot()
    def load_video(self):
        options = QFileDialog.Options()
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Load Video", "", "Video Files (*.mp4 *.avi)", options=options)
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_dropdown.clear()
            self.frame_dropdown.addItems([str(f) for f in range(self.total_frames)])
            self.update_frame()

    @pyqtSlot()
    def next_frame(self):
        if self.cap is not None and self.current_frame + 1 < self.total_frames:
            self.current_frame += 10
            self.update_frame()

    @pyqtSlot()
    def previous_frame(self):
        if self.cap is not None and self.current_frame - 1 >= 0:
            self.current_frame -= 10
            self.update_frame()

    def update_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to format suitable for Qt
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(self.width, self.height, Qt.KeepAspectRatio)
            self.video_frame_label.setPixmap(QPixmap.fromImage(p))

            # Update the status bar
            current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            total_time = self.total_frames / self.cap.get(cv2.CAP_PROP_FPS)
            self.status_bar.setText(
                f"Frame: {self.current_frame} / {self.total_frames}\n"
                f"Time: {current_time:.2f}s / {total_time:.2f} s"
            )

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.next_frame()
        elif event.key() == Qt.Key_Left:
            self.previous_frame()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = VideoPlayer()
    ex.show()
    sys.exit(app.exec_())
