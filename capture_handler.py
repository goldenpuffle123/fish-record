import PySpin
from PySide6.QtCore import (
    Signal,
    QObject
)
import numpy as np

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