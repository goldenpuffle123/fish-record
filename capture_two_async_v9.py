# Using experimental v2 CameraDriver
# SharedMemory architecture with ffmpeg-python wrapper
# Clean FFmpeg integration with better error handling
# Cleaned up all unused functions
from capture_handler_v3 import BufferedCaptureHandler
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
import threading
# Removed: import ffmpeg  # Not using ffmpeg-python wrapper anymore
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] - %(message)s')

class ImageLabel(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.target_size: QSize = self.size()
    
    def resizeEvent(self, event: QResizeEvent) -> None:
        self.target_size: QSize = self.size()
        super().resizeEvent(event)


class VideoWriterProcess(QObject):
    """Video writer using QProcess in dedicated thread (fixed Qt threading issues)"""
    
    def __init__(
            self,
            cam_idx: int,
            capture_handler: BufferedCaptureHandler,
            frame_rate: float,
            date_time: str,
            output_dir: str = "synced_videos"
        ) -> None:
        super().__init__()
        
        self.cam_idx = cam_idx
        self.capture_handler = capture_handler
        self.frame_rate = frame_rate
        self.date_time = date_time
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Frame info queue for writing
        self.write_queue = queue.Queue(maxsize=100)
        self.is_recording = False
        self.dropped_frames = 0
        
        # QProcess will be created in the writer thread to avoid Qt threading issues
        self.ffmpeg_process = None
        self.writer_thread = None
    
    def _get_ffmpeg_args(self):
        """Get FFmpeg arguments optimized for both recording and playback"""
        frame_shape = self.capture_handler.frame_shape
        h, w = frame_shape
        
        return [
            'ffmpeg',
            '-y',                          # Overwrite output without asking
            '-f', 'rawvideo',              # Input format: raw video frames
            '-vcodec', 'rawvideo',         # Input codec: uncompressed video
            '-s', f'{w}x{h}',              # Frame size (width x height)
            '-pix_fmt', 'gray',            # Input pixel format: 8-bit grayscale
            '-r', str(self.frame_rate),    # Input frame rate
            '-i', '-',                     # Read from stdin
            
            # NVIDIA NVENC H.264 Parameters (optimized for playback)
            '-c:v', 'h264_nvenc',          # NVIDIA H.264 encoder
            '-preset', 'p6',               # Quality/speed preset (was p1, p3 worked in v5)
            '-rc', 'vbr',               # Bitrate type (working from v5)
            '-cq', '20',          # Constant quality mode (best quality approach)
            '-b:v', '25M',                 # Target bitrate (working from v5)
            '-maxrate', '35M',             # Max bitrate (working from v5)
            '-bufsize', '45M',             # Buffer size (working from v5)
            '-gpu', '0',                   # Use GPU 0
            '-profile:v', 'main',          # Compatibility profile
            '-surfaces', '8',              # NVENC surfaces (working from v5)
            '-vf', 'format=yuv420p',       # Video filter (working from v5)
            '-color_range', 'pc',          # Color range (working from v5)
            '-colorspace','bt709',         # Color space (working from v5)
            '-f', 'mp4',                   # Output container format
            '-g', '300',
            '-keyint_min', '100',
            '-bf', '2',           # B-frames for temporal compression
            '-refs', '2',         # More reference frames
            '-spatial_aq', '1',   # Spatial adaptive quantization
            '-temporal_aq', '0',  # Temporal adaptive quantization
            '-rc-lookahead', '5', # Lookahead for better encoding decisions
            self._get_filename()
        ]
    
    def start_recording(self):
        """Start the recording process in a dedicated thread"""
        self.is_recording = True
        
        # Start writer thread which will create and manage QProcess
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        
        logging.info(f"Started FFmpeg recording camera {self.cam_idx} to {self._get_filename()}")
    
    def queue_frame_info(self, frame_id: int, buffer_idx: int):
        """Queue frame info for writing (called by SyncHandler)"""
        if not self.is_recording:
            return
            
        try:
            self.write_queue.put_nowait((frame_id, buffer_idx))
        except queue.Full:
            self.dropped_frames += 1
            if self.dropped_frames % 50 == 0:
                logging.warning(f"Cam {self.cam_idx} writer: Dropped {self.dropped_frames} frames")
    
    def _writer_loop(self):
        """Background thread that creates QProcess and handles writing"""
        frames_written = 0
        
        # Create QProcess in this thread to avoid Qt threading issues
        self.ffmpeg_process = QProcess()
        self.ffmpeg_process.setProgram("ffmpeg")
        self.ffmpeg_process.setArguments(self._get_ffmpeg_args()[1:])  # Skip 'ffmpeg' executable name
        
        try:
            # Start FFmpeg process
            self.ffmpeg_process.start()
            
            # Wait for process to start
            if not self.ffmpeg_process.waitForStarted(5000):  # 5 second timeout
                logging.error(f"FFmpeg process failed to start for camera {self.cam_idx}")
                return
            
            logging.info(f"FFmpeg process started for camera {self.cam_idx}")
            
            # Main writing loop
            while self.is_recording or not self.write_queue.empty():
                try:
                    frame_id, buffer_idx = self.write_queue.get(timeout=0.1)
                    
                    # Check if QProcess is still running
                    if self.ffmpeg_process.state() != QProcess.ProcessState.Running:
                        logging.error(f"FFmpeg process died for camera {self.cam_idx}")
                        break
                    
                    # Get frame data from shared memory (v9 approach)
                    frame_data = self.capture_handler.get_frame_raw(buffer_idx)
                    
                    # Write to QProcess (v5 approach)
                    if frame_data:
                        bytes_written = self.ffmpeg_process.write(frame_data)
                        if bytes_written == -1:
                            logging.error(f"Failed to write frame to FFmpeg for camera {self.cam_idx}")
                            break
                        
                        frames_written += 1
                        
                        # Flush periodically (like v5)
                        if frames_written % 10 == 0:
                            self.ffmpeg_process.waitForBytesWritten(100)  # 100ms timeout
                        
                except queue.Empty:
                    if not self.is_recording:
                        break
                    continue
                except Exception as e:
                    logging.error(f"Error in writer loop for camera {self.cam_idx}: {e}")
                    break
            
        except Exception as e:
            logging.error(f"Failed to start FFmpeg for camera {self.cam_idx}: {e}")
        finally:
            # Cleanup QProcess
            self._cleanup_process()
            
        logging.info(f"Writer loop finished for cam {self.cam_idx}, wrote {frames_written} frames")
    
    def _cleanup_process(self):
        """Clean up the FFmpeg process"""
        if self.ffmpeg_process and self.ffmpeg_process.state() == QProcess.ProcessState.Running:
            logging.info(f"Closing write channel for cam {self.cam_idx}...")
            self.ffmpeg_process.closeWriteChannel()
            
            if not self.ffmpeg_process.waitForFinished(10000):  # 10 second timeout
                logging.warning(f"FFmpeg process for cam {self.cam_idx} didn't finish, terminating...")
                self.ffmpeg_process.terminate()
                if not self.ffmpeg_process.waitForFinished(3000):  # 3 second timeout
                    logging.error(f"Force killing FFmpeg process for cam {self.cam_idx}")
                    self.ffmpeg_process.kill()
    
    def stop_recording(self):
        """Stop recording and cleanup"""
        logging.info(f"Stopping recording for camera {self.cam_idx}...")
        self.is_recording = False
        
        # Wait for writer thread to finish
        if hasattr(self, 'writer_thread') and self.writer_thread.is_alive():
            logging.info(f"Waiting for writer thread to finish (cam {self.cam_idx})...")
            self.writer_thread.join(timeout=5.0)
        
        logging.info(f"Stopped recording cam {self.cam_idx}. Dropped {self.dropped_frames} frames.")
    
    def _get_filename(self) -> str:
        return f"{self.output_dir}/data_{self.date_time}_cam-{self.cam_idx}.mp4"


class SyncHandler(QObject):
    # Only emit for display purposes, much less frequently
    display_frame_ready: Signal = Signal(np.ndarray, np.ndarray)

    def __init__(self, cam_list: list[PySpin.Camera], frame_rate: float):
        super().__init__()
        self.cam_list = cam_list

        self.capture_handlers: list[BufferedCaptureHandler] = []
        self.video_writers: list[VideoWriterProcess] = []

        self.cam_info: list[tuple] = CameraDriver.get_resolution_list(cam_list)
        date_time = datetime.now().strftime('%y%m%d-%H%M%S')

        for cam_idx in range(2):
            frame_shape = self.cam_info[cam_idx][::-1]  # (height, width) instead of (width, height)
            handler = BufferedCaptureHandler(cam_idx, frame_shape)

            self.capture_handlers.append(handler)
            
            # Connect frame capture to sync processing
            handler.frame_captured.connect(self.receive_frame_info)
            self.cam_list[cam_idx].RegisterEventHandler(handler)
            
            # Create video writer for this camera
            writer = VideoWriterProcess(cam_idx, handler, frame_rate, date_time)
            self.video_writers.append(writer)

        # Store frame info for sync
        self.frame_info = [queue.Queue(), queue.Queue()]
        
        # Display and sync control
        self.display_every_n = 10  # Display every 5th frame (20Hz at 100Hz capture)
        self.frame_counter = 0
        self.warmup_frames = 300

    def start(self):
        """Start camera acquisition and recording"""
        # Start video writers first
        for writer in self.video_writers:
            writer.start_recording()
            
        # Start cameras (order matters!)
        self.cam_list[1].BeginAcquisition()
        self.cam_list[0].BeginAcquisition()
        
        logging.info("Started synchronized capture and recording")

    def receive_frame_info(self, cam_idx: int, frame_id: int, buffer_idx: int):
        """Handle incoming frame info"""
        self.frame_info[cam_idx].put((frame_id, buffer_idx))
        self.sync()

    def sync(self):
        """Synchronize frames and manage writing/display with improved sync checking"""
        if self.frame_info[0].empty() or self.frame_info[1].empty():
            return
        
        # Peek at frame IDs without removing from queue
        id0, buffer_idx0 = self.frame_info[0].queue[0]
        id1, buffer_idx1 = self.frame_info[1].queue[0]

        if id0 == id1:
            # Frames are synchronized
            
            # Check if both writers can accept frames (prevent individual drops)
            if self.frame_counter > self.warmup_frames:
                can_write_0 = not self.video_writers[0].write_queue.full()
                can_write_1 = not self.video_writers[1].write_queue.full()
                
                if can_write_0 and can_write_1:
                    # Both can write - queue for both
                    self.video_writers[0].queue_frame_info(id0, buffer_idx0)
                    self.video_writers[1].queue_frame_info(id1, buffer_idx1)
            
            # Only emit for display every Nth frame
            if self.frame_counter % self.display_every_n == 0:
                im0 = self.capture_handlers[0].get_frame(buffer_idx0)
                im1 = self.capture_handlers[1].get_frame(buffer_idx1)
                self.display_frame_ready.emit(im0, im1)
            
            # Remove processed info
            self.frame_info[0].get()
            self.frame_info[1].get()
            self.frame_counter += 1
            
        else:
            # Handle frame drops
            if id0 < id1:
                self.frame_info[0].get()
                logging.warning(f"Camera sync: Dropped cam 0 frame {id0}")
            else:
                self.frame_info[1].get() 
                logging.warning(f"Camera sync: Dropped cam 1 frame {id1}")

    def stop(self):
        """Stop everything cleanly"""
        logging.info("Stopping synchronized capture...")
        
        # Stop cameras first
        self.cam_list[0].EndAcquisition()
        self.cam_list[1].EndAcquisition()
        self.cam_list[0].UnregisterEventHandler(self.capture_handlers[0])
        self.cam_list[1].UnregisterEventHandler(self.capture_handlers[1])
        
        # Stop video writers
        for writer in self.video_writers:
            writer.stop_recording()
        
        # Clean up shared memory buffers
        for handler in self.capture_handlers:
            handler.cleanup()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_cd()

        # Misc "global" variables
        self.display_counter = 0
        self.cam_info: list[tuple] = [
            CameraDriver.get_resolution(self.cd.cam_list[0]),
            CameraDriver.get_resolution(self.cd.cam_list[1]),
        ]
        
        self.launch_captures()

    def setup_ui(self):
        self.setWindowTitle("capture two async videos - FFmpeg-Python v9")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.video_labels = [ImageLabel(), ImageLabel()]
        layout.addWidget(self.video_labels[0])
        layout.addWidget(self.video_labels[1])
        self.resize(QSize(1000, 400))

    def launch_captures(self):
        """Launch the unified sync handler"""
        self.sync_handler = SyncHandler(self.cd.cam_list, self.cd.ACQUISITION_FRAME_RATE)
        self.sync_handler.display_frame_ready.connect(self.display_images)
        self.sync_handler.start()

        self.time_start = time.perf_counter()
        self.time_count = 0

    def display_images(self, im0: np.ndarray, im1: np.ndarray):
        """Handle display updates (called only every Nth frame)"""
        
        # Status updates
        if time.perf_counter() - self.time_start > self.time_count:
            logging.info(f"Ran {self.time_count:.1f} s")
            self.time_count += 20  # Update every 20 seconds
        
        # Update display
        for im_num, im in enumerate((im0, im1)):
            qim = QImage(im, self.cam_info[im_num][0], self.cam_info[im_num][1], self.cam_info[im_num][0], QImage.Format.Format_Grayscale8)
            scaled_pm = QPixmap.fromImage(qim).scaled(
                self.video_labels[im_num].target_size,
                aspectMode=Qt.AspectRatioMode.KeepAspectRatio,
                mode=Qt.TransformationMode.FastTransformation
            )
            self.video_labels[im_num].setPixmap(scaled_pm)

    def setup_cd(self):
        self.cd = CameraDriver()
        self.cd.initialize_cameras()
        self.cd.set_config_all()
        self.cd.set_config_sync_two()

    def closeEvent(self, event: QCloseEvent):
        """Override to ensure clean shutdown"""
        logging.info("Shutting down application...")
        self.sync_handler.stop()
        self.cd.release_all()
        logging.info("Application shutdown complete.")
        event.accept()

def main():
    app = QApplication()

    window = MainWindow()
    window.show()

    try:
        app.exec()
    finally:
        logging.info("Application exiting...")

if __name__ == "__main__":
    main()
