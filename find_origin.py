from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QScrollArea
)

from PySide6.QtGui import (
    QPixmap,
    QImage,
    Qt,
    QPainter,
    QPen,
    QKeyEvent,
    QMouseEvent,
    QWheelEvent
)

from PySide6.QtCore import (
    QPoint,
    Signal
)
import cv2
import numpy as np

import select_dialog
from pathlib import Path


class KeypointWidget(QWidget):
    closed = Signal()
    def __init__(self, path: str, origin_points: list, idx: int) -> None:
        super().__init__()
        self.fact = 1
        self.idx = idx
        self.pointer = QPoint(0, 0)

        frame: np.ndarray = self._get_first_frame(path)

        im = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1], QImage.Format.Format_Grayscale8)
        cal_pix = QPixmap.fromImage(im)

        self.cal_pix_scaled = cal_pix.scaled(
            cal_pix.size()*self.fact,
            aspectMode=Qt.AspectRatioMode.KeepAspectRatio
        )

        self.origin_points = origin_points

        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.resize(self.cal_pix_scaled.size())
        #self.showMaximized()

    def sizeHint(self):
        """Return the size of the scaled pixmap for proper scrollbar sizing"""
        return self.cal_pix_scaled.size()

    def minimumSizeHint(self):
        """Return the minimum size needed to display the scaled pixmap"""
        return self.cal_pix_scaled.size()

    @staticmethod
    def _get_first_frame(path: str) -> np.ndarray:
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError(f"Error reading video file: {path}")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        step = 1
        step_dict = {
            Qt.Key.Key_Left: (-step, 0),
            Qt.Key.Key_Right: (step, 0),
            Qt.Key.Key_Up: (0, -step),
            Qt.Key.Key_Down: (0, step)
        }
        #fine adjustments
        if event.key() in step_dict:
            dx, dy = step_dict[event.key()]
            self.pointer += QPoint(dx, dy)
            self.update()
        #save point
        elif event.key() == Qt.Key.Key_Return:
            self.origin_points[self.idx] = (self.pointer.x()/self.fact,
                                               self.pointer.y()/self.fact)
            self.close()
            self.closed.emit()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.pointer = QPoint(int(event.position().x()),
                                  int(event.position().y()))
            self.update()
            
    
    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.cal_pix_scaled)
        painter.setPen(QPen(Qt.GlobalColor.red, 1))
        painter.drawEllipse(self.pointer, 1, 1)

class KeypointScroller(QScrollArea):
    def __init__(self, path: str, origin_points: list, idx: int):
        super().__init__()
        widget = KeypointWidget(path, origin_points, idx)
        self.setWidget(widget)
        widget.closed.connect(self.close)
        self.setWidgetResizable(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setWindowTitle(f"cam {idx}")
        self.showMaximized()

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() == Qt.KeyboardModifier.AltModifier:
            self.horizontalScrollBar().wheelEvent(event)
        else:
            self.verticalScrollBar().wheelEvent(event)
        

if __name__ == "__main__":
    app = QApplication([])

    try:
        projection_matrices_path = "cal_images_250807-1400_5cm/projection_matrices.npz"
        projection_matrices = np.load(projection_matrices_path)
        #np.load(select_dialog.get_filepath("Select projection matrices file", filter="*.npz"))
    except OSError:
        print("Projection matrices file not found.")
        quit()
    P0 = projection_matrices["P0"]
    P1 = projection_matrices["P1"]

    cali_matrices = np.load("cal_images_250807-1400_5cm/stereo_matrices.npz")
    
    for abc in range(2):
        origin_points = [None, None]
        test_videos = ["synced_videos/data_250807-144447_cam-0_5cm.mp4",
                    "synced_videos/data_250807-144447_cam-1_5cm.mp4"]
        #np.load(select_dialog.get_filepaths("Select test videos", filter="*.mp4", pair=True))

        windows = [
            KeypointScroller(test_videos[i], origin_points, i)
            for i in (0,1)
        ]

        windows[0].show()
        windows[1].show()

        app.exec()


        if None not in origin_points:
            #print(f"Origin points selected: {origin_points}")

            origin_points[0] = cv2.undistortPoints(np.array(origin_points[0]), cali_matrices["K0"], cali_matrices["dist0"], P=cali_matrices["K0"])
            origin_points[1] = cv2.undistortPoints(np.array(origin_points[1]), cali_matrices["K1"], cali_matrices["dist1"], P=cali_matrices["K1"])
   
            points_4d = cv2.triangulatePoints(
                P0, P1,
                np.array((origin_points[0])).T,
                np.array((origin_points[1])).T
            )
            points_3d = (points_4d[:3] / points_4d[3]).T
            print(f"Triangulated origin point: {points_3d[0]}")
        # np.save(Path(projection_matrices).parent / "water_point.npy", points_3d[0])