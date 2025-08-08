from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QLabel,
    QWidget,
    QStackedLayout
)

from PySide6.QtGui import (
    QPixmap,
    QImage,
    QScreen,
    Qt,
    QPainter,
    QPen,
    QKeyEvent,
    QMouseEvent,
    QCloseEvent
)

from PySide6.QtCore import (
    QPoint
)


class KeypointWidget(QWidget):
    def __init__(self, im: QImage):
        super().__init__()

        self.fact = 0.5

        cal_pix = QPixmap.fromImage(im)

        self.cal_pix_scaled = cal_pix.scaled(
            cal_pix.size()*self.fact,
            aspectMode=Qt.AspectRatioMode.KeepAspectRatio
        )

        self.pointer = QPoint(0, 0)
        self.saved_points = []

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, event: QKeyEvent):
        #fine adjustments
        if event.key() == Qt.Key.Key_Left:
            self.pointer.setX(self.pointer.x() - 1)
        elif event.key() == Qt.Key.Key_Right:
            self.pointer.setX(self.pointer.x() + 1)
        elif event.key() == Qt.Key.Key_Up:
            self.pointer.setY(self.pointer.y() - 1)
        elif event.key() == Qt.Key.Key_Down:
            self.pointer.setY(self.pointer.y() + 1)
        #save point
        elif event.key() == Qt.Key.Key_Return:
            self.saved_points.append(
                (self.pointer.x()/self.fact,
                self.pointer.y()/self.fact)
            )
            print(self.saved_points)
        #delete point and go back to prev
        elif event.key() == Qt.Key.Key_Delete:
            if self.saved_points:
                self.saved_points.pop()
                if self.saved_points:
                    self.pointer.setX(int(self.saved_points[-1][0]))
                    self.pointer.setY(int(self.saved_points[-1][1]))
                else:
                    self.pointer.setX(0)
                    self.pointer.setY(0)

        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self.pointer.setX(int(pos.x()))
            self.pointer.setY(int(pos.y()))
            self.update()
            
    
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.cal_pix_scaled)
        painter.setPen(QPen(Qt.GlobalColor.red, 1))
        painter.drawEllipse(self.pointer, 2, 2)

class KeypointWindow(QMainWindow):
    def __init__(self, path: str):
        super().__init__()
        self.calibration_widget = KeypointWidget(path)
        self.setCentralWidget(self.calibration_widget)
        self.move(0,0)
        self.showMaximized()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.open_button = QPushButton("Open window")
        self.setCentralWidget(self.open_button)
        self.open_button.clicked.connect(self.open_calibration)
        self.cal_window = None

    def open_calibration(self):
        self.cal_window = KeypointWindow()
        self.cal_window.show()

    def closeEvent(self, event: QCloseEvent):
        if self.cal_window is not None:
            self.cal_window.close()
        event.accept()
        



if __name__ == "__main__":
    app = QApplication([])
    windows = [
        KeypointWindow("data_250730-170548_cam-0.mp4"),
        KeypointWindow("data_250730-170548_cam-1.mp4")
    ]
    app.exec()
