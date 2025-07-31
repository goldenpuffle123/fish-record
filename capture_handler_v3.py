import PySpin
from PySide6.QtCore import (
    Signal,
    QObject
)
import numpy as np
from multiprocessing import shared_memory
import threading

class BufferedCaptureHandler(PySpin.ImageEventHandler, QObject):
    # Signal now only carries frame info, not the actual image data
    frame_captured: Signal = Signal(int, int, int)  # cam_idx, frame_id, buffer_idx

    def __init__(self, idx: int, frame_shape: tuple):
        PySpin.ImageEventHandler.__init__(self)
        QObject.__init__(self)
        self.idx = idx
        
        # 3 buffers: allows more flexibility for read/write operations
        self.buffers: list[shared_memory.SharedMemory] = []
        for i in range(3):
            name = f"cam-{idx}_buf-{i}"
            shm = shared_memory.SharedMemory(create=True, size=int(np.prod(frame_shape)), name=name)
            self.buffers.append(shm)
        
        self.frame_shape = frame_shape
        self.current_write = 0
        self.write_lock = threading.RLock()

    def OnImageEvent(self, image: PySpin.Image):
        im_np: np.ndarray = image.GetNDArray()
        im_id: int = image.GetFrameID()
        
        with self.write_lock:
            # Write to current buffer
            np_view = np.ndarray(self.frame_shape, dtype=im_np.dtype, 
                               buffer=self.buffers[self.current_write].buf)
            np_view[:] = im_np
            
            buffer_idx = self.current_write
            # Cycle through 3 buffers for next write
            self.current_write = (self.current_write + 1) % 3
        
        self.frame_captured.emit(self.idx, im_id, buffer_idx)
        image.Release()
    
    def get_frame(self, buffer_idx: int) -> np.ndarray:
        """Return frame copy from specified buffer"""
        with self.write_lock:
            np_view = np.ndarray(self.frame_shape, dtype=np.uint8, buffer=self.buffers[buffer_idx].buf)
            return np_view.copy()  # Copy to avoid memory issues
    
    def get_frame_raw(self, buffer_idx: int) -> bytes:
        """Read raw frame data from specified buffer for direct writing"""
        with self.write_lock:
            np_view = np.ndarray(self.frame_shape, dtype=np.uint8, buffer=self.buffers[buffer_idx].buf)
            return np_view.copy().tobytes()
    def cleanup(self):
        """Clean up shared memory buffers"""
        for buffer in self.buffers:
            try:
                buffer.close()
                buffer.unlink()
            except:
                pass
