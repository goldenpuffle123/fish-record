# Using experimental v2 CameraDriver
# Added black screen write for dropped frames
# Merged VideoWriterThreads
# SharedMemory architecture for better performance - DoubleBuffer version
from capture_handler_v3 import DoubleBufferCaptureHandler
from camera_driver_v2 import CameraDriver
from PySide6.QtCore import (
    QSize,
    QObject,
    QThread,
    Signal,
    Qt,
    QProcess,
    QByteArray
)
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QResizeEvent,
    QCloseEvent
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
import cv2
from datetime import datetime
import os
import typing
import time

class ImageLabel(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.target_size: QSize = self.size()
    
    def resizeEvent(self, event: QResizeEvent) -> None:
        self.target_size: QSize = self.size()
        super().resizeEvent(event)



class VideoWriterProcess(QProcess):
    def __init__(
            self,
            idx: int,
            fr: float,
            cam_info: list[tuple],
            date_time: str,
            output_dir: str = "synced_videos"
        ) -> None:
        super().__init__()
        self.save_queue = queue.Queue(maxsize=200)  # Limit queue size to prevent memory buildup
        self.cam_idx = idx
        w, h = cam_info
        self.date_time = date_time
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self._configure_ffmpeg(w, h, fr)
        self.dropped_frames = 0

    def _configure_ffmpeg(self,
        w: int,
        h: int,
        fr: float
    ):
        # THIS IS A FUCKING MESS!!! FIX!!!
        self.setProgram("ffmpeg")
        self.setArguments([
            '-y',                          # Overwrite output without asking
            '-f', 'rawvideo',              # Input format: raw video frames
            '-vcodec', 'rawvideo',         # Input codec: uncompressed video
            '-s', f'{w}x{h}',              # Frame size (width x height)
            '-pix_fmt', 'gray',            # Input pixel format: 8-bit grayscale
            '-r', str(fr),                 # Input frame rate
            '-i', '-',                     # Read from stdin
            
            # NVIDIA NVENC H.264 Parameters
            '-c:v', 'h264_nvenc',          # NVIDIA H.264 encoder
            '-preset', 'p3',               # Quality/speed preset
            '-rc', 'vbr_hq',               # Bitrate type
            '-b:v', '15M',                 # Target bitrate (adjust based on resolution)
            '-gpu', '0',                   # Use GPU 0
            '-profile:v', 'main',          # Compatibility profile
            '-tune', 'ull',                # Mode ultra-low-latency
            '-zerolatency', '1',           # Zero latency encoding
            '-rc-lookahead', '0',          # Disable lookahead for lowest latency
            '-no-scenecut', '1',           # Disable scene cut detection
            '-forced-idr', '1',
            '-surfaces', '8',
            '-vf', 'format=yuv420p',
            '-color_range', 'pc',
            '-colorspace','bt709',
            '-f', 'mp4',                   # Output container format
            self._get_filename()
        ])
    
    def _get_filename(self) -> str:
        return f"{self.output_dir}/data_{self.date_time}_cam-{self.cam_idx}_IMPORTANT.mp4"

    

    def queue_frame(self, im: np.ndarray):
        try:
            self.save_queue.put_nowait(im.tobytes())
            self._process_queue()
        except queue.Full:
            # If queue is full, drop this frame to prevent memory buildup
            self.dropped_frames += 1
            if self.dropped_frames % 50 == 0:  # Print every 50 dropped frames
                print(f"VideoWriter {self.cam_idx}: Dropped {self.dropped_frames} frames due to full queue")

    def _process_queue(self):
        while self.state() == QProcess.ProcessState.Running:
            try:
                frame_data = self.save_queue.get_nowait()
                self.write(frame_data)
                self.waitForBytesWritten()
            except queue.Empty:
                break
    
    def _process_remaining(self):
        while not self.save_queue.empty():
            try:
                frame_data = self.save_queue.get_nowait()
                self.write(frame_data)
            except queue.Empty:
                break
    def stop(self):
        if self.state() == QProcess.ProcessState.Running:
            self._process_remaining()  # Flush remaining frames
            self.closeWriteChannel()
            self.waitForFinished()


class SyncThread(QObject):
    next_images: Signal = Signal(np.ndarray, np.ndarray)

    def __init__(self, cam_list: list[PySpin.Camera]):
        super().__init__()

        self.cam_list = cam_list
        self.capture_handlers: list[DoubleBufferCaptureHandler] = []
        self.cam_info: list[tuple] = [
            CameraDriver.get_resolution(self.cam_list[0]),
            CameraDriver.get_resolution(self.cam_list[1]),
        ]

        # Create handlers with frame shape information
        for cam_idx in range(2):
            frame_shape = (self.cam_info[cam_idx][1], self.cam_info[cam_idx][0])  # (height, width)
            self.capture_handlers.append(DoubleBufferCaptureHandler(cam_idx, frame_shape))
            self.capture_handlers[cam_idx].frame_captured.connect(self.receive_frame_metadata)
            self.cam_list[cam_idx].RegisterEventHandler(self.capture_handlers[cam_idx])

        # Store frame metadata instead of full frames
        self.frame_metadata = [queue.Queue(), queue.Queue()]

    def start(self):
        self.cam_list[1].BeginAcquisition() # Order is VERY important
        self.cam_list[0].BeginAcquisition()

    def save_placeholders(self):
        im0: np.ndarray = np.zeros((self.cam_info[0][::-1]), dtype=np.uint8)
        im1: np.ndarray = np.zeros((self.cam_info[1][::-1]), dtype=np.uint8)
        self.next_images.emit(im0, im1)

    def receive_frame_metadata(self, cam_idx: int, frame_id: int, buffer_idx: int):
        self.frame_metadata[cam_idx].put((frame_id, buffer_idx))
        self.sync()

    def sync(self):
        if self.frame_metadata[0].empty() or self.frame_metadata[1].empty():
            return
        
        # Peek at frame IDs without removing from queue
        id0, buffer_idx0 = self.frame_metadata[0].queue[0]
        id1, buffer_idx1 = self.frame_metadata[1].queue[0]

        if id0 == id1:
            # Frames are synchronized - retrieve actual frame data from shared memory
            im0 = self.capture_handlers[0].get_frame(buffer_idx0)
            im1 = self.capture_handlers[1].get_frame(buffer_idx1)
            
            self.next_images.emit(im0, im1)
            self.frame_metadata[0].get()
            self.frame_metadata[1].get()
        else:
            self.save_placeholders()
            if id0 < id1:
                self.frame_metadata[0].get()
                print(f"dropped cam 0 frame {id0}")
            else:
                self.frame_metadata[1].get()
                print(f"dropped cam 1 frame {id1}")
            

    def stop(self):
        self.cam_list[0].EndAcquisition()
        self.cam_list[1].EndAcquisition()
        self.cam_list[0].UnregisterEventHandler(self.capture_handlers[0])
        self.cam_list[1].UnregisterEventHandler(self.capture_handlers[1])
        
        # Clean up shared memory buffers
        for handler in self.capture_handlers:
            handler.cleanup()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_cd()

        # Misc "global" variables
        self.frame_counter = 0
        self.display_every_n = 3 # Controls display update rate
        self.cam_info: list[tuple] = [
            CameraDriver.get_resolution(self.cd.cam_list[0]),
            CameraDriver.get_resolution(self.cd.cam_list[1]),
        ]
        self.date_time = datetime.now().strftime('%y%m%d-%H%M%S')
        
        self.launch_save_processes()
        self.launch_captures()

    def setup_ui(self):
        self.setWindowTitle("capture two async videos")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.video_labels = [ImageLabel(), ImageLabel()]
        layout.addWidget(self.video_labels[0])
        layout.addWidget(self.video_labels[1])
        self.resize(QSize(1000, 400))

    def launch_save_processes(self):
        self.save_processes: list[VideoWriterProcess] = []
        for cam_idx in range(2):
            self.save_processes.append(
                VideoWriterProcess(
                    cam_idx,
                    self.cd.ACQUISITION_FRAME_RATE,
                    self.cam_info[cam_idx],
                    self.date_time
                )
            )
            self.save_processes[cam_idx].start()

    def launch_captures(self):
        self.sync_thread = SyncThread(self.cd.cam_list)
        self.sync_thread.next_images.connect(self.display_images)
        self.sync_thread.start()

        self.time_start = time.time()
        self.time_count = 0

    def display_images(self, im0: np.ndarray, im1: np.ndarray):
        if self.frame_counter >= 100:    # Starts after a warmup
            self.save_processes[0].queue_frame(im0)
            self.save_processes[1].queue_frame(im1)

        if time.time()-self.time_start > self.time_count:
            print(f"elapsed: {self.time_count:2f} s")
            self.time_count += 20 # Update every 20 secs
        
        

        if self.frame_counter % self.display_every_n == 0:
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

    def setup_cd(self):
        self.cd = CameraDriver()
        self.cd.initialize_cameras()
        self.cd.set_config_all()
        self.cd.set_config_sync_two()

    def closeEvent(self, event: QCloseEvent): # Overrides QMainWindow method
        self.sync_thread.stop()
        self.cd.release_all()
        for process in self.save_processes:
            process.stop()
            process.deleteLater()

        
        
        event.accept() # let the window close

def main():
    app = QApplication()

    window = MainWindow()

    window.show()

    app.exec()

if __name__ == "__main__":
    main()