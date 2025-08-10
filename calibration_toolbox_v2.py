import numpy as np
import cv2
from pathlib import Path
import select_dialog
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
                 file_prefix: str = "cam-") -> None:

        self.calibration_dir = Path(select_dialog.get_dir("Select calibration images directory"))

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

    
    def modify_image(self, image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def calibrate_camera(self,
                         cam_idx: int,
                         save_info: bool = False) -> tuple[np.ndarray, np.ndarray] | None:
        
        """Gets intrinsic matrix (from v1)"""
        W = self.pts_w
        H = self.pts_h

        # Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        
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

                # cv2.imshow(f"imgs {cam_idx}", img_display)
                logging.info(f"Found in {img_path.name}")
            else:
                logging.info(f"Not found in {img_path.name}")
            # cv2.waitKey()

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
            return mtx, dist
        else:
            logging.error(f"Error in calibrating {cam_idx}")
        cv2.destroyAllWindows()

    def calibrate_stereo(self,
                         K0: np.ndarray,
                         dist0: np.ndarray,
                         K1: np.ndarray,
                         dist1: np.ndarray,
                         save_info: bool = False
                         ) -> tuple[np.ndarray,
                                    np.ndarray,
                                    np.ndarray,
                                    np.ndarray,
                                    np.ndarray,
                                    np.ndarray,
                                    np.ndarray,
                                    np.ndarray] | None:
        W = self.pts_w
        H = self.pts_h

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # Note: converted to mm 
        objp = np.zeros((H*W,3), np.float32)
        objp[:,:2] = np.mgrid[0:W,0:H].T.reshape(-1,2)
        objp *= self.square_size_mm

        

        # Look for images in calibration images directory of the specified file prefix, camera indexing, and file type
        cam_0_paths = sorted(self.calibration_dir.glob(f"{self.file_prefix}0*.png"))
        cam_1_paths = sorted(self.calibration_dir.glob(f"{self.file_prefix}1*.png"))

        image_size = None

        images_0 = []
        images_1 = []

        for img_0_path, img_1_path in zip(cam_0_paths, cam_1_paths):
            _im = cv2.imread(str(img_0_path), cv2.IMREAD_GRAYSCALE)
            images_0.append(_im)
            _im = cv2.imread(str(img_1_path), cv2.IMREAD_GRAYSCALE)
            images_1.append(_im)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)


        objpoints = []  # Points for this camera
        imgpoints_0 = []
        imgpoints_1 = []

        for img_0, img_1 in zip(images_0, images_1):
            if img_0 is None or img_1 is None:
                logging.warning(f"Failed to load one of the image pairs")
                continue




            # Find the chess board corners
            ret0, corners0 = cv2.findChessboardCorners(img_0, (W,H))
            ret1, corners1 = cv2.findChessboardCorners(img_1, (W,H))

            # If not found, try histogram eq
            if not ret0:
                img_0 = self.modify_image(img_0)
                ret0, corners0 = cv2.findChessboardCorners(img_0, (W,H))
            if not ret1:
                img_1 = self.modify_image(img_1)
                ret1, corners1 = cv2.findChessboardCorners(img_1, (W,H))

            # If found, add object points, image points (after refining them)
            if ret0 and ret1:
                if image_size is None:
                    image_size = img_0.shape[::-1]

                

                corners0 = cv2.cornerSubPix(img_0, corners0, (11,11), (-1,-1), criteria)
                

                corners1 = cv2.cornerSubPix(img_1, corners1, (11,11), (-1,-1), criteria)
                

                # Draw and display the corners
                cv2.drawChessboardCorners(img_0, (W, H), corners0, ret0)
                cv2.drawChessboardCorners(img_1, (W, H), corners1, ret1)

                img_0_display = img_0.copy()
                img_1_display = img_1.copy()

                """ cv2.imshow(f"imgs {0}", cv2.resize(img_0_display, None, fx=0.8, fy=0.8))
                cv2.imshow(f"imgs {1}", cv2.resize(img_1_display, None, fx=0.8, fy=0.8))

                cv2.waitKey() """

                objpoints.append(objp)
                imgpoints_0.append(corners0)
                imgpoints_1.append(corners1)
            else:
                logging.warning("No chessboards detected in pair!")

        cv2.destroyAllWindows()

        if not imgpoints_0 or not imgpoints_1:
            logging.warning("No valid image points found.")
            return

        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
        ret, K0, dist0, K1, dist1, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints_0,
            imgpoints_1,
            K0, dist0,
            K1,
            dist1,
            image_size,
            criteria=criteria,
            flags=stereocalibration_flags
        )
        if ret:
            if save_info:
                npz_save = str(self.calibration_dir)+f"/stereo_matrices"
                np.savez(npz_save, K0=K0, dist0=dist0, K1=K1, dist1=dist1, R=R, T=T, E=E, F=F)
                logging.info(f"Saved stereo calibration matrices in {npz_save}")
            return K0, dist0, K1, dist1, R, T, E, F
        
    def get_projection_matrices(self,
                                K0: np.ndarray,
                                K1: np.ndarray,
                                R: np.ndarray,
                                T: np.ndarray,
                                save_info: bool) -> tuple[np.ndarray, np.ndarray]:
        """Get projection matrices P0 and P1 from intrinsic matrices K0, K1 and extrinsic parameters R, T"""
        P0 = K0 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P1 = K1 @ np.hstack((R, T))

        if save_info:
            npz_save = str(self.calibration_dir) + f"/projection_matrices"
            np.savez(npz_save, P0=P0, P1=P1)
            logging.info(f"Saved projection matrices in {npz_save}")
        return P0, P1



if __name__ == "__main__":
    ct = CalibrationToolbox(pts_w=6,
                            pts_h=4,
                            square_size_mm=10)
    
    ct.calibrate_camera(0, save_info=True)
    ct.calibrate_camera(1, save_info=True)

    filenames = [f"{ct.calibration_dir}/cam-{i}.npz" for i in (0, 1)]
    data_0 = np.load(filenames[0])
    data_1 = np.load(filenames[1])

    K0, dist0 = data_0['mtx'], data_0['dist']
    K1, dist1 = data_1['mtx'], data_1['dist']

    K0, dist0, K1, dist1, R, T, E, F = ct.calibrate_stereo(K0, dist0, K1, dist1, save_info=True)

    P0, P1 = ct.get_projection_matrices(K0, K1, R, T, save_info=True)