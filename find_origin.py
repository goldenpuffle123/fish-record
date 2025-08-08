from PySide6.QtWidgets import (
    QApplication,
    QWidget,
)

from PySide6.QtGui import (
    QPixmap,
    QImage,
    Qt,
    QPainter,
    QPen,
    QKeyEvent,
    QMouseEvent,
)

from PySide6.QtCore import (
    QPoint
)
import cv2
import numpy as np

import select_dialog
from pathlib import Path

import decord


class KeypointWidget(QWidget):

    def __init__(self, path: str, origin_points: list, idx: int):
        super().__init__()
        self.fact = 1
        self.idx = idx
        self.pointer = QPoint(0, 0)

        frame: np.ndarray = self._get_first_frame(path)

        im = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format.Format_Grayscale8)
        cal_pix = QPixmap.fromImage(im)

        self.cal_pix_scaled = cal_pix.scaled(
            cal_pix.size()*self.fact,
            aspectMode=Qt.AspectRatioMode.KeepAspectRatio
        )

        self.origin_points = origin_points

        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.move(0,0)
        self.showMaximized()

    @staticmethod
    def _get_first_frame(path: str) -> np.ndarray:
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError(f"Error reading video file: {path}")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def keyPressEvent(self, event: QKeyEvent):
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

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.pointer = QPoint(int(event.position().x()),
                                  int(event.position().y()))
            self.update()
            
    
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.cal_pix_scaled)
        painter.setPen(QPen(Qt.GlobalColor.red, 1))
        painter.drawEllipse(self.pointer, 2, 2)



if __name__ == "__main__":
    app = QApplication([])

    try:
        stereo_matrices = np.load(select_dialog.get_filepath("Select stereo matrices file", filter="*.npz"))
    except OSError:
        print("Stereo matrices file not found.")
        quit()
    P0 = stereo_matrices["P0"]
    P1 = stereo_matrices["P1"]

    origin_points = [None, None]
    test_videos = select_dialog.get_filepaths("Select test videos", filter="*.mp4", pair=True)
    windows = [
        KeypointWidget(test_videos[0], origin_points, 0),
        KeypointWidget(test_videos[1], origin_points, 1)
    ]
    
    windows[0].show()
    windows[1].show()

    app.exec()

    if None not in origin_points:
        print(f"Origin points selected: {origin_points}")
        points_4d = cv2.triangulatePoints(
            P0, P1,
            np.array((origin_points[0])).T,
            np.array((origin_points[1])).T
        )
        points_3d = (points_4d[:3] / points_4d[3]).T
        print(f"Triangulated origin point: {points_3d[0]}")
        # np.save(Path(stereo_matrices).parent / "origin_point.npy", points_3d[0])