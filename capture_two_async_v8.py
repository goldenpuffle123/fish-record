# Using experimental v2 CameraDriver
# SharedMemory architecture with ffmpeg-python wrapper
# Clean FFmpeg integration with better error handling
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
import ffmpeg

class ImageLabel(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.target_size: QSize = self.size()
    
    def resizeEvent(self, event: QResizeEvent) -> None:
        self.target_size: QSize = self.size()
        super().resizeEvent(event)


class VideoWriter(QObject):
    """VideoWriter using ffmpeg-python wrapper for cleaner configuration"""
    
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
        self.write_queue = queue.Queue(maxsize=300)
        self.is_recording = False
        self.dropped_frames = 0
        
        # FFmpeg process
        self.ffmpeg_process = None
        self.writer_thread = None
        
    def start_recording(self):
        """Start the recording process using ffmpeg-python"""
        self.is_recording = True
        
        # Get frame dimensions
        frame_shape = self.capture_handler.frame_shape
        h, w = frame_shape
        
        try:
            # Configure ffmpeg stream using ffmpeg-python
            input_stream = ffmpeg.input('pipe:', 
                format='rawvideo',
                pix_fmt='gray',
                s=f'{w}x{h}',
                r=self.frame_rate
            )
            
            # Configure output with NVENC
            output_stream = ffmpeg.output(
                input_stream,
                self._get_filename(),
                vcodec='h264_nvenc',
                preset='fast',
                rc='vbr',
                **{
                    'b:v': '15M',
                    'maxrate': '20M',
                    'bufsize': '30M',
                    'profile:v': 'main',
                    'level': '4.1',
                    'pix_fmt': 'yuv420p',
                    'movflags': '+faststart'  # Optimize for streaming
                }
            )
            
            # Start the ffmpeg process
            self.ffmpeg_process = ffmpeg.run_async(
                output_stream,
                pipe_stdin=True,
                pipe_stdout=True,
                pipe_stderr=True,
                overwrite_output=True
            )
            
            # Start writer thread
            self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self.writer_thread.start()
            
            print(f"Started FFmpeg recording camera {self.cam_idx} to {self._get_filename()}")
            
        except Exception as e:
            print(f"Failed to start FFmpeg for camera {self.cam_idx}: {e}")
            self.is_recording = False
            if self.ffmpeg_process:
                try:
                    self.ffmpeg_process.terminate()
                except:
                    pass
    
    def queue_frame_info(self, frame_id: int, buffer_idx: int):
        """Queue frame info for writing (called by SyncHandler)"""
        if not self.is_recording:
            return
            
        try:
            self.write_queue.put_nowait((frame_id, buffer_idx))
        except queue.Full:
            if self.dropped_frames % 50 == 0:
                print(f"Camera {self.cam_idx}: Dropped {self.dropped_frames} frames (writer too slow)")
            self.dropped_frames += 1
    
    def _writer_loop(self):
        """Background thread that reads from shared memory and writes to FFmpeg"""
        while self.is_recording or not self.write_queue.empty():
            try:
                # Get frame info with timeout
                frame_id, buffer_idx = self.write_queue.get(timeout=0.1)
                
                # Read frame directly from shared memory
                frame_data = self.capture_handler.get_frame_raw(buffer_idx)
                
                # Write to FFmpeg
                if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                    self.ffmpeg_process.stdin.write(frame_data)
                    # Don't flush every frame for better performance
                    if frame_id % 10 == 0:  # Flush every 10 frames
                        self.ffmpeg_process.stdin.flush()
                else:
                    print(f"FFmpeg process died for camera {self.cam_idx}")
                    self._check_ffmpeg_error()
                    break
                    
            except queue.Empty: # No frames found to write, just continue
                continue
            except BrokenPipeError:
                print(f"FFmpeg pipe broken for camera {self.cam_idx}")
                self._check_ffmpeg_error()
                break
            except Exception as e:
                print(f"Error in writer loop for camera {self.cam_idx}: {e}")
                break
    
    def _check_ffmpeg_error(self):
        """Check FFmpeg stderr for error messages"""
        if self.ffmpeg_process and self.ffmpeg_process.stderr:
            try:
                stderr_output = self.ffmpeg_process.stderr.read(1024).decode('utf-8', errors='ignore')
                if stderr_output:
                    print(f"FFmpeg error (cam {self.cam_idx}): {stderr_output}")
            except:
                pass
    
    def stop_recording(self):
        """Stop recording and cleanup"""
        self.is_recording = False
        
        # Wait for writer thread to finish
        if self.writer_thread:
            self.writer_thread.join(timeout=5.0)
        
        # Close FFmpeg gracefully
        if self.ffmpeg_process:
            try:
                # Close stdin to signal end of input
                if self.ffmpeg_process.stdin:
                    self.ffmpeg_process.stdin.close()
                
                # Wait for process to finish
                return_code = self.ffmpeg_process.wait(timeout=15.0)
                if return_code != 0:
                    print(f"FFmpeg process for camera {self.cam_idx} exited with code {return_code}")
                    self._check_ffmpeg_error()
                    
            except Exception as e:
                print(f"Error stopping FFmpeg for camera {self.cam_idx}: {e}")
                try:
                    self.ffmpeg_process.terminate()
                    self.ffmpeg_process.wait(timeout=5.0)
                except:
                    self.ffmpeg_process.kill()
        
        print(f"Stopped recording camera {self.cam_idx}. Dropped {self.dropped_frames} frames.")
    
    def _get_filename(self) -> str:
        return f"{self.output_dir}/data_{self.date_time}_cam-{self.cam_idx}_FFMPEG.mp4"
    
    def get_stats(self) -> dict:
        """Get recording statistics"""
        return {
            'cam_idx': self.cam_idx,
            'dropped_frames': self.dropped_frames,
            'queue_size': self.write_queue.qsize(),
            'is_recording': self.is_recording,
            'ffmpeg_alive': self.ffmpeg_process and self.ffmpeg_process.poll() is None
        }


class SyncHandler(QObject):
    # Only emit for display purposes, much less frequently
    display_frame_ready: Signal = Signal(np.ndarray, np.ndarray)

    def __init__(self, cam_list: list[PySpin.Camera], frame_rate: float):
        super().__init__()
        self.cam_list = cam_list

        self.capture_handlers: list[BufferedCaptureHandler] = []
        self.video_writers: list[VideoWriter] = []

        self.cam_info: list[tuple] = [
            CameraDriver.get_resolution(self.cam_list[0]),
            CameraDriver.get_resolution(self.cam_list[1]),
        ]
        date_time = datetime.now().strftime('%y%m%d-%H%M%S')

        for cam_idx in range(2):
            frame_shape = self.cam_info[cam_idx][::-1]  # (height, width) instead of (width, height)
            handler = BufferedCaptureHandler(cam_idx, frame_shape)

            self.capture_handlers.append(handler)
            
            # Connect frame capture to sync processing
            handler.frame_captured.connect(self.receive_frame_info)
            self.cam_list[cam_idx].RegisterEventHandler(handler)
            
            # Create video writer for this camera
            writer = VideoWriter(cam_idx, handler, frame_rate, date_time)
            self.video_writers.append(writer)

        # Store frame info for sync
        self.frame_info = [queue.Queue(), queue.Queue()]
        
        # Display and sync control
        self.display_every_n = 5  # Display every 5th frame (20Hz at 100Hz capture)
        self.frame_counter = 0
        self.warmup_frames = 100
        self.sync_dropped_frames = 0
        
        # Statistics
        self.stats_timer = 0
        self.last_stats_time = time.perf_counter()

    def start(self):
        """Start camera acquisition and recording"""
        # Start video writers first
        for writer in self.video_writers:
            writer.start_recording()
            
        # Start cameras (order matters!)
        self.cam_list[1].BeginAcquisition()
        self.cam_list[0].BeginAcquisition()
        
        print("Started synchronized capture and recording")

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
            # Frames are synchronized!
            
            # Check if both writers can accept frames (prevent individual drops)
            can_write_0 = not self.video_writers[0].write_queue.full()
            can_write_1 = not self.video_writers[1].write_queue.full()
            
            if can_write_0 and can_write_1 and self.frame_counter >= self.warmup_frames:
                # Both can write - queue for both
                self.video_writers[0].queue_frame_info(id0, buffer_idx0)
                self.video_writers[1].queue_frame_info(id1, buffer_idx1)
            elif self.frame_counter >= self.warmup_frames:
                # At least one can't write - drop from both to maintain sync
                self.sync_dropped_frames += 1
                if self.sync_dropped_frames % 100 == 0:
                    print(f"Sync: Dropped {self.sync_dropped_frames} frames (queue full)")
            
            # Only emit for display every Nth frame
            if self.frame_counter % self.display_every_n == 0:
                im0 = self.capture_handlers[0].get_frame(buffer_idx0)
                im1 = self.capture_handlers[1].get_frame(buffer_idx1)
                self.display_frame_ready.emit(im0, im1)
            
            # Remove processed info
            self.frame_info[0].get()
            self.frame_info[1].get()
            self.frame_counter += 1
            
            # Print stats periodically
            self._update_stats()
            
        else:
            # Handle frame drops
            if id0 < id1:
                self.frame_info[0].get()
                print(f"Camera sync: Dropped cam 0 frame {id0}")
            else:
                self.frame_info[1].get() 
                print(f"Camera sync: Dropped cam 1 frame {id1}")

    def _update_stats(self):
        """Print periodic statistics"""
        current_time = time.perf_counter()
        if current_time - self.last_stats_time > 30:  # Every 30 seconds
            self.last_stats_time = current_time
            
            print("\n=== Recording Statistics ===")
            print(f"Total frames processed: {self.frame_counter}")
            print(f"Sync dropped frames: {self.sync_dropped_frames}")
            
            for writer in self.video_writers:
                stats = writer.get_stats()
                print(f"Camera {stats['cam_idx']}: "
                      f"Queue: {stats['queue_size']}/300, "
                      f"Dropped: {stats['dropped_frames']}, "
                      f"FFmpeg: {'OK' if stats['ffmpeg_alive'] else 'DEAD'}")
            print("===========================\n")

    def stop(self):
        """Stop everything cleanly"""
        print("Stopping synchronized capture...")
        
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
        
        # Final stats
        print(f"\nFinal Statistics:")
        print(f"Total frames: {self.frame_counter}")
        print(f"Sync drops: {self.sync_dropped_frames}")
        for writer in self.video_writers:
            print(f"Camera {writer.cam_idx} drops: {writer.dropped_frames}")


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
        self.setWindowTitle("capture two async videos - FFmpeg-Python v8")
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
        """Launch the unified sync handler that manages everything"""
        self.sync_handler = SyncHandler(self.cd.cam_list, self.cd.ACQUISITION_FRAME_RATE)
        self.sync_handler.display_frame_ready.connect(self.display_images)
        self.sync_handler.start()

        self.time_start = time.perf_counter()
        self.time_count = 0

    def display_images(self, im0: np.ndarray, im1: np.ndarray):
        """Handle display updates (called only every Nth frame)"""
        self.display_counter += 1
        
        # Status updates
        if time.perf_counter() - self.time_start > self.time_count:
            print(f"Display update: {self.time_count:.1f}s - Frame #{self.display_counter}")
            self.time_count += 20  # Update every 20 seconds
        
        # Update display
        for im_num, im in enumerate((im0, im1)):
            qim = QImage(im, self.cam_info[im_num][0], self.cam_info[im_num][1], QImage.Format.Format_Grayscale8)
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
        print("Shutting down application...")
        self.sync_handler.stop()
        self.cd.release_all()
        print("Application shutdown complete.")
        event.accept()

def main():
    app = QApplication()

    window = MainWindow()
    window.show()

    try:
        app.exec()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Application exiting...")

if __name__ == "__main__":
    main()
