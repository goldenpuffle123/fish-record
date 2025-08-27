# For capturing checkboard images.
# New version uses imageevent and driver v2
from datetime import datetime
from camera_driver_v2 import CameraDriver
from PySide6.QtCore import (
    QSize,
    QObject,
    QThread,
    Signal,
    Qt,
    QRunnable,
    QThreadPool
)
from PySide6.QtGui import (
    QImage,
    QPixmap
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QHBoxLayout,
    QWidget
)
import numpy as np
import PySpin
import queue
from pathlib import Path
import cv2

class CaptureHandler(PySpin.ImageEventHandler, QObject): # Need to subclass QObject as well for Signals
    next_image = Signal(np.ndarray, int, int)

    def __init__(self, idx: int):
        PySpin.ImageEventHandler.__init__(self)
        QObject.__init__(self)
        self.idx = idx
    
    def OnImageEvent(self, image: PySpin.Image):
        im_np: np.ndarray = image.GetNDArray()
        im_id: int = image.GetFrameID()
        self.next_image.emit(im_np, self.idx, im_id)
        image.Release()

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.target_size = self.size()
    
    def resizeEvent(self, event):
        self.target_size = self.size()
        super().resizeEvent(event)

class SaveThread(QRunnable):
    def __init__(self,
        im_np: np.ndarray,
        cam_idx: int,
        save_dir: Path,
        save_count: int
    ):
        super().__init__()
        self.im_np = im_np
        self.save_dir = save_dir
        self.cam_idx = cam_idx
        self.save_count = save_count

    def _get_filename(self) -> str:
        fn = f"cam-{self.cam_idx}-{self.save_count:02d}.png"
        return str(self.save_dir / fn)

    def run(self):
        cv2.imwrite(self._get_filename(), self.im_np)
        

class SyncThread(QObject):
    next_images = Signal(np.ndarray, np.ndarray)

    def __init__(self, cam_list: list[PySpin.Camera], save_dir: Path):
        super().__init__()
        self.cam_list = cam_list
        self.save_dir = save_dir
        self.save = False
        self.save_count = 0
        self.buffers = [queue.Queue() for _ in range(len(cam_list))]
        self.thread_pool = QThreadPool.globalInstance()
        
        # Simplified camera handler setup
        self.capture_handlers = []
        for cam_idx, camera in enumerate(cam_list):
            handler = CaptureHandler(cam_idx)
            handler.next_image.connect(self.receive_image)
            camera.RegisterEventHandler(handler)
            self.capture_handlers.append(handler)

    def start(self):
        self.cam_list[1].BeginAcquisition()
        self.cam_list[0].BeginAcquisition()

    def receive_image(self, im_np: np.ndarray, idx: int, id: int):
        self.buffers[idx].put((im_np, id))
        self.sync()

    def sync(self):
        if self.buffers[0].empty() or self.buffers[1].empty():
            return
        
        im0, id0 = self.buffers[0].queue[0]
        im1, id1 = self.buffers[1].queue[0]

        if id0 == id1:
            self.next_images.emit(im0.copy(), im1.copy())
            self.buffers[0].get()
            self.buffers[1].get()
            if self.save:
                self.save_images(im0, im1)
                self.save = False
        elif id0 < id1:
            self.buffers[0].get()
            print(f"dropped cam 1 frame {id0}")
        else:
            self.buffers[1].get()
            print(f"dropped cam 0 frame {id1}")
    
    def handle_save_request(self):
        self.save = True

    def save_images(self, im0, im1):
        workers: list[SaveThread] = []
        for cam_idx, im in enumerate((im0, im1)):
            workers.append(
                SaveThread(im, cam_idx, self.save_dir, self.save_count)
            )
            self.thread_pool.start(workers[cam_idx])
        self.save_count += 1
        print(f"Img pair {self.save_count} saved")

    def stop(self):
        self.cam_list[0].EndAcquisition()
        self.cam_list[1].EndAcquisition()
        self.cam_list[0].UnregisterEventHandler(self.capture_handlers[0])
        self.cam_list[1].UnregisterEventHandler(self.capture_handlers[1])


class MainWindow(QMainWindow):
    DISPLAY_EVERY_N_FRAMES = 3
    SAVE_MODE = True  # Can either use as general viewing or saving

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_cd()

        self.frame_counter = 0
        self.cam_info: list[tuple] = CameraDriver.get_resolution_list(self.cd.cam_list)
        self.date_time = datetime.now().strftime('%y%m%d-%H%M')
        self.save_dir = Path(f"cal_images_{self.date_time}")
        if self.SAVE_MODE:
            self.save_dir.mkdir(exist_ok=True)

        self.launch_captures()

    def setup_ui(self):
        self.setWindowTitle("Capture calibration pairs (enter to save)")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)


        self.video_labels = [ImageLabel(), ImageLabel()]
        layout.addWidget(self.video_labels[0])
        layout.addWidget(self.video_labels[1])

        self.showMaximized()

    def launch_captures(self):
        self.sync_thread = SyncThread(self.cd.cam_list, self.save_dir)
        self.sync_thread.next_images.connect(self.display_images)
        self.sync_thread.start()

    def display_images(self, im0, im1):

        if self.frame_counter % self.DISPLAY_EVERY_N_FRAMES == 0:
            for im_num, im in enumerate((im0, im1)):
                qim = QImage(im, self.cam_info[im_num][0], self.cam_info[im_num][1], QImage.Format.Format_Grayscale8)
                pm = QPixmap.fromImage(qim)
                scaled_pm = pm.scaled(
                    self.video_labels[im_num].target_size,
                    aspectMode=Qt.AspectRatioMode.KeepAspectRatio,
                    mode=Qt.TransformationMode.FastTransformation
                )
                self.video_labels[im_num].setPixmap(scaled_pm)

        self.frame_counter += 1
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return and self.SAVE_MODE:
            self.sync_thread.handle_save_request()

    def setup_cd(self):
        self.cd = CameraDriver()
        self.cd.initialize_cameras()
        self.cd.set_config_all()
        self.cd.set_config_sync_two()

    def closeEvent(self, event): # Overrides QMainWindow method

        self.sync_thread.stop()

        self.cd.release_all()
        
        event.accept() # let the window close

def main():
    app = QApplication([])

    window = MainWindow()

    window.show()

    app.exec()

if __name__ == "__main__":
    main()