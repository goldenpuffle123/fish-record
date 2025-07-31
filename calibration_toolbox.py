import numpy as np
import cv2
from pathlib import Path
from PySide6.QtWidgets import QFileDialog, QApplication
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class CalibrationToolbox:
    def __init__(self,
                 pts_w: int,
                 pts_h: int,
                 square_size_mm: float,
                 file_prefix: str = "cam-",
                 calibration_dir: str = None) -> None:
        
        if calibration_dir is None:
            self.calibration_dir = Path(self.get_folder())
        else:
            self.calibration_dir = Path(calibration_dir)

        if not self.calibration_dir.exists():
            logging.error("Calibration images directory not found!")
            raise FileNotFoundError(f"Calibration images directory {self.calibration_dir} not found!")
        
        logging.info(f"Using directory {self.calibration_dir}")

        self.file_prefix = file_prefix  # For glob function

        self.pts_w = pts_w    # Number of internal points on checkboard
        self.pts_h = pts_h
        self.square_size_mm = square_size_mm

        self.num_cameras = 2
        self.cam_matrices = [None]*self.num_cameras

    def get_folder(self) -> str:
        app = QApplication.instance()
        if app is None:                     # In case implemented into complete GUI
            app = QApplication([])
        folder = QFileDialog.getExistingDirectory(
            parent=None,
            caption="Select calibration images directory"
        )
        if folder=="":
            logging.error("Select dialog cancelled")
            raise FileNotFoundError("Select dialog cancelled")

        return folder
    
    def modify_image(self, image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def calibrate_camera(self, cam_idx: int, save_info: bool = True) -> None:
        W = self.pts_w
        H = self.pts_h

        # Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        objpoints = []  # Points for this camera
        imgpoints = []

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # Note: converted to mm 
        objp = np.zeros((H*W,3), np.float32)
        objp[:,:2] = np.mgrid[0:W,0:H].T.reshape(-1,2) * self.square_size_mm

        

        # Look for images in calibration images directory of the specified file prefix, camera indexing, and file type
        pattern = f"{self.file_prefix}{cam_idx}*.png"
        images = list(self.calibration_dir.glob(pattern))
        
        image_size = None

        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logging.warning(f"Failed to load {img_path.name}")
                continue

            img_mod = img.copy()

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(img_mod, (W,H))

            # If not found, try histogram eq
            if not ret:
                img_mod = self.modify_image(img_mod)
                ret, corners = cv2.findChessboardCorners(img_mod, (W,H))

            # If found, add object points, image points (after refining them)
            if ret:
                if image_size is None:
                    image_size = img_mod.shape[::-1]

                objpoints.append(objp)
        
                corners2 = cv2.cornerSubPix(img_mod, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)
        
                # Draw and display the corners
                
                cv2.drawChessboardCorners(img, (W, H), corners2, ret)

                img_display = img.copy()
                max_w, max_h = 1500, 1200  # adjust as needed for screen
                h, w = img_display.shape[:2]
                scale = min(max_w / w, max_h / h, 1.0)
                new_size = (int(w * scale), int(h * scale))
                img_display = cv2.resize(img_display, new_size)

                cv2.imshow(f"imgs {cam_idx}", img_display)
                logging.info(f"Found in {img_path.name}")
            else:
                logging.info(f"Not found in {img_path.name}")
            cv2.waitKey(500)

        if not objpoints or not imgpoints:
            logging.error(f"No valid detections for camera {cam_idx}")
            cv2.destroyAllWindows()
            return
 
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
        if ret:
            self.cam_matrices[cam_idx] = (mtx, dist)
            if save_info:
                npz_save = str(self.calibration_dir)+f"/{self.file_prefix}{cam_idx}"
                np.savez(npz_save, mtx=mtx, dist=dist)
                logging.info(f"Saved mtx and dist in {npz_save}")
        else:
            logging.error(f"Error in calibrating {cam_idx}")
        cv2.destroyAllWindows()
        



if __name__ == "__main__":
    ct = CalibrationToolbox(pts_w=6,
                            pts_h=4,
                            square_size_mm=10)
    ct.calibrate_camera(0)
    ct.calibrate_camera(1)
    