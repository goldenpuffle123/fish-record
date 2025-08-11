import cv2
import numpy as np


class RefractionCalibration:
    def __init__(self, cali_path: str, upload_path: str = "") -> None:

        self.cali_matrices: dict[str, np.ndarray] = np.load(cali_path)
        self.K = [self.cali_matrices["K0"], self.cali_matrices["K1"]]
        self.dist = [self.cali_matrices["dist0"], self.cali_matrices["dist1"]]
        self.inv_K = [np.linalg.inv(self.K[0]), np.linalg.inv(self.K[1])] # Compute inv ahead of time
        self.R_stereo = self.cali_matrices["R"]
        self.T_stereo = self.cali_matrices["T"].flatten()

        self.ray_params_names = [
            # Camera origins
            "cam0_origin",
            "cam1_origin",
            # For getting camera 1 origin relative to camera 0
            "R_cam_to_board",
            "T_cam_to_board",
            # Water plane parameters
            "water_plane_normal",
            "water_plane_d",
            # For getting final transformation from camera to box
            "R_final_cam_to_box",
            "R_final_cam_to_box_inv",
            "T_final_cam_to_box"
        ]

        if upload_path:
            ray_params = np.load(upload_path)
            for name in self.ray_params_names:
                if name not in ray_params.files:
                    raise ValueError(f"Parameter '{name}' not found in uploaded file.")
                setattr(self, name, ray_params[name])
        else:
            for name in self.ray_params_names:
                setattr(self, name, None)

        # Refraction indexes
        self.n1 = 1.0003
        self.n2 = 1.333
        self.n_ratio = self.n1 / self.n2
        

    def setup_stereo_geometry(self) -> None:
        """Setup stereo camera geometry from calibration matrices."""
        # Camera 0 at origin
        self.cam0_origin = np.array([0.0, 0.0, 0.0])
        
        # Camera 1 position relative to camera 0
        self.cam1_origin = -self.R_stereo.T @ self.T_stereo
        print("cam1_origin:", self.cam1_origin)

    def get_pnp(self, chessboard_path: str, W: int, H: int, square_size_mm: float) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Solve PnP for camera 0 to table."""
        # Chessboard flat on table. We assume this is parallel to the water surface.
        img = cv2.imread(chessboard_path, cv2.IMREAD_GRAYSCALE)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        objp = np.zeros((H*W,3), np.float32)
        objp[:,:2] = np.mgrid[0:W,0:H].T.reshape(-1,2)
        objp *= square_size_mm
        ret, corners = cv2.findChessboardCorners(img, (W,H))
        if not ret:
            raise ValueError("Chessboard corners not found in the image.")
        corners2 = cv2.cornerSubPix(img, corners, (11,11), (-1,-1), criteria)
        imgpoints = corners2
        ret, rvec, tvec = cv2.solvePnP(objp, imgpoints, self.K[0], self.dist[0])
        if not ret:
            raise ValueError("Error solving PnP.")

        if ret:
            # Convert rvec to rotation matrix
            self.R_cam_to_board, _ = cv2.Rodrigues(rvec)
            self.T_cam_to_board = tvec.flatten()
            return self.R_cam_to_board, self.T_cam_to_board
        return None, None

    def get_water_parameters(self, point: np.ndarray) -> tuple[np.ndarray, float]:
        """Get water plane normal vector and water plane distance"""
        normal = self.R_cam_to_board[:, 2] # Third column in rotation matrix
        normal /= np.linalg.norm(normal)

        dist = np.dot(normal, point)
        self.water_plane_d = dist

        # As convention we are keeping this as output of solvepnp (facing away from camera)
        self.water_plane_normal = normal
        
        return normal, dist

    def unproject_pixel(self, p_undistorted: np.ndarray, idx: int) -> np.ndarray:
        """Get 3D ray direction given a pixel in camera view"""
        p_homogeneous = np.append(p_undistorted, 1.0) # [x, y, 1]

        # Un-project by multiplying with the inverse of the intrinsic matrix
        ray_dir = self.inv_K[idx] @ p_homogeneous
        # Normalize
        return ray_dir/np.linalg.norm(ray_dir)

    def intersect_ray_with_plane(self, ray_origin: np.ndarray, ray_dir: np.ndarray) -> np.ndarray | None:
        """
        Finds the intersection of a 3D ray with a plane.
        
        Args:
            ray_origin (np.ndarray): The 3D origin point of the ray.
            ray_dir (np.ndarray): The 3D direction vector of the ray.
            plane_normal (np.ndarray): The normal vector of the plane.
            plane_d (float): The scalar distance of the plane from the origin.
            
        Returns:
            np.ndarray or None: The 3D intersection point, or None if no intersection.
        """
        # As from https://math.stackexchange.com/a/100447

        denom = np.dot(self.water_plane_normal, ray_dir)

        # Check if the ray does not intersect plane i.e., parallel
        if np.abs(denom) < 1e-8:
            return None

        t = (self.water_plane_d - np.dot(self.water_plane_normal, ray_origin)) / denom
        
        # We are only interested in intersections in front of the camera
        if t >= 0:
            point = ray_origin + t * ray_dir
            return point
        return None

    def refract_vector(self, incident_vec: np.ndarray) -> np.ndarray | None:
        """
        Calculates the refracted vector using the vector form of Snell's Law.
        
        Args:
            incident_vec (np.ndarray): The normalized incident vector.
            normal_vec (np.ndarray): The normalized surface normal vector.
            n1 (float): Refractive index of the first medium (e.g., air).
            n2 (float): Refractive index of the second medium (e.g., water).
            
        Returns:
            np.ndarray or None: The refracted direction vector, or None for TIR.
        """
        I = incident_vec
        N = self.water_plane_normal

        cos1 = -np.dot(N, I)

        # cos1 must be positive, which it will be if n is the normal vector that points from the surface toward the side
        # where the light is coming from, the region with refraction index n1. If cos1 is negative, then n points to the side
        # without the light, so start over with n replaced by its negative.
        # Note from https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
        if cos1 < 0:
            N = -N
            cos1 = -np.dot(N, I)

        # Check for Total Internal Reflection
        sin2_sq = self.n_ratio**2 * (1.0 - cos1**2)
        if sin2_sq > 1.0:
            return None # TIR occurred

        cos2 = np.sqrt(1.0 - sin2_sq)

        refracted_vec = self.n_ratio * I + (self.n_ratio * cos1 - cos2) * N
        refracted_vec /= np.linalg.norm(refracted_vec)
        return refracted_vec

    def find_closest_point_between_lines(self, o0: np.ndarray, d0: np.ndarray, o1: np.ndarray, d1: np.ndarray) -> np.ndarray:
        """
        Finds the midpoint of the shortest line segment connecting two 3D lines.
        This serves as the best-estimate intersection point.
        
        Args:
            o0, d0 (np.ndarray): Origin and direction vector for line 1.
            o1, d1 (np.ndarray): Origin and direction vector for line 2.
            
        Returns:
            np.ndarray: The 3D point of closest approach.
        """
        threshold = 1e-8
        d0 = d0 / np.linalg.norm(d0)
        d1 = d1 / np.linalg.norm(d1)
        # Line perpendicular to both lines
        n = np.cross(d0, d1)

        if np.linalg.norm(n) < threshold:
            # Lines are parallel - use midpoint method
            w = o0 - o1
            t0 = np.dot(w, d0)
            t1 = np.dot(w, d1)
            
            closest_on_line1 = o0 - t0 * d0
            closest_on_line2 = o1 + t1 * d1
            
            return (closest_on_line1 + closest_on_line2) / 2.0

        n0 = np.cross(d0, n)
        n1 = np.cross(d1, n)

        denom0 = np.dot(d0, n1)
        denom1 = np.dot(d1, n0)

        if abs(denom0) < threshold or abs(denom1) < threshold:
            return (o0 + o1) / 2.0

        # Calculate the points on each line that form the shortest segment
        c0 = o0 + d0 * (np.dot((o1 - o0), n1) / denom0)
        c1 = o1 + d1 * (np.dot((o0 - o1), n0) / denom1)

        return (c0 + c1) / 2.0

    def undistort_pixel(self, p_coords: np.ndarray, idx: int) -> np.ndarray:
        """Undistorts pixel coordinates using the camera's distortion parameters."""
        # Comments: assume p_coords comes directly from YOLO, as xy = kp.xy.numpy().
        # Each keypoint is given as xy[0][j] for keypoint j --> np array of length 2
        # Input/output form for function is a bit unclear, see
        # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga55c716492470bfe86b0ee9bf3a1f0f7e
        result = cv2.undistortPoints(p_coords, self.K[idx], self.dist[idx], P=self.K[idx])
        return result

    def correct_underwater_point(self, p_raw_0: np.ndarray, p_raw_1: np.ndarray) -> np.ndarray | None:
        """
        Main workflow: Convert stereo pixel coordinates to corrected 3D underwater point.
        
        Args:
            p_raw_0: [x, y] pixel coordinates in camera 0
            p_raw_1: [x, y] pixel coordinates in camera 1

        Returns:
            underwater_point: [x, y, z] corrected 3D coordinates
        """
        
        # Step 1: Undistort pixels
        p_undist_0 = self.undistort_pixel(p_raw_0, 0)
        p_undist_1 = self.undistort_pixel(p_raw_1, 1)

        # Step 2: Unproject to 3D ray directions
        ray0_dir = self.unproject_pixel(p_undist_0, 0)
        ray1_dir_cam1 = self.unproject_pixel(p_undist_1, 1)
        
        # Transform camera 1 ray to world coordinates
        ray1_dir = self.R_stereo.T @ ray1_dir_cam1

        # Step 3: Find intersections with water surface
        water_intersection0 = self.intersect_ray_with_plane(self.cam0_origin, ray0_dir)
        water_intersection1 = self.intersect_ray_with_plane(self.cam1_origin, ray1_dir)

        if water_intersection0 is None or water_intersection1 is None:
            return None
        
        # Step 4: Apply Snell's law refraction
        refracted_ray0 = self.refract_vector(ray0_dir)
        refracted_ray1 = self.refract_vector(ray1_dir)

        if refracted_ray0 is None or refracted_ray1 is None:
            return None  # Total internal reflection
        
        # Step 5: Triangulate underwater point using refracted rays
        underwater_point = self.find_closest_point_between_lines(
            water_intersection0, refracted_ray0,
            water_intersection1, refracted_ray1
        )
        return underwater_point

    def setup_transformation(self, p_box_origin_cam, p_box_x_axis_cam, p_box_y_axis_cam = None):
        """
        Setup parameters for the final transformation from camera to box.
        
        p_box_y_axis_cam is not required as it is calculated from dot product.
        """
        self.T_final_cam_to_box = p_box_origin_cam
        x_axis_vec = p_box_x_axis_cam - p_box_origin_cam
        x_axis = x_axis_vec / np.linalg.norm(x_axis_vec)
        z_axis = self.water_plane_normal
        y_axis_raw = np.cross(z_axis, x_axis)
        y_axis = y_axis_raw / np.linalg.norm(y_axis_raw) # y-axis points down
        """ y_axis_vec = p_box_y_axis_cam - p_box_origin_cam
        y_axis = y_axis_vec / np.linalg.norm(y_axis_vec) """
        self.R_final_cam_to_box = np.column_stack((x_axis, y_axis, z_axis)) # Construct transformation matrix
        self.R_final_cam_to_box_inv = np.linalg.inv(self.R_final_cam_to_box) # Precompute inverse

    def transform_point(self, point):
        """Transform a point from camera coordinates to box coordinates."""
        # Transform point
        return self.R_final_cam_to_box_inv @ (point - self.T_final_cam_to_box)
    
    def save_parameters(self, path: str = ""):
        params_to_save = {}
        for name in self.ray_params_names:
            value = getattr(self, name, None)
            if value is None:
                raise ValueError(f"Parameter '{name}' is not initialized and cannot be saved.")
            params_to_save[name] = value

        # Save to npz
        np.savez(path, **params_to_save)
        print(f"Saved parameters to {path}")

if __name__ == "__main__":
    rc = RefractionCalibration("cal_images_250807-1641_10cm/stereo_matrices.npz", upload_path="cal_images_250807-1641_10cm/ray_parameters.npz")
    # rc.setup_stereo_geometry()
    # rc.get_pnp("cal_images_250807-1641_10cm/cam-0-00.png", 6, 4, 10)

    # Get these from find_origin.py
    # Note: already undistorted
    """ p_water_cam = np.array([45.31909385197085, -22.482463212596368, 274.023306783117])  # using top right """
    p_water_cam = np.array([-47.167812398340914, 52.11593132229903, 445.0398179347797])
    # rc.get_water_parameters(p_water_cam)

    """ p_box_origin_cam = np.array([-7.43308086, -24.59866516, 266.73127528]) # tl
    p_box_x_axis_cam = np.array([ 47.52253882, -24.38689605, 267.31237704]) # tr
    p_box_y_axis_cam = np.array([ -7.48576695,  30.48156394, 266.76362464]) # bl """
    p_box_origin_cam = np.array([-49.638713274534176, -45.97083631512408, 430.57195239144664]) # tl
    p_box_x_axis_cam = np.array([50.416291444974846, -46.48186324869854, 435.3122079279627]) # tr
    p_box_y_axis_cam = np.array([-48.621162654825525, 54.19308586553376, 429.52718188283563]) # bl

    # rc.setup_transformation(p_box_origin_cam, p_box_x_axis_cam, p_box_y_axis_cam)
    """ br_corner = np.array([47.26931526610344, 30.348581650670887, 266.60577051482846])
    
    tl_water = np.array([-4.371592286697936, -20.99286077087117, 291.5036169620433])
    center_water = np.array([19.340094309309077, 3.143551594866998, 298.07068311511085])
    tr_water = np.array([43.20729824531479, -20.501389250425298, 290.0414210602504])
    water_1 = np.array([38.38620655896026, -15.834747097777955, 296.22866926976496])
    water_2 = np.array([24.87567695813077, -8.428278710198931, 297.0490228170133])
    water_3 = np.array([39.630995430465525, -6.726793676224169, 298.25170986568327]) """
    water_1 = np.array([-2.2376416019022405, 4.457238843732878, 503.41024057177475])
    water_2 = np.array([-7.289524693064915, 34.78150165622865, 503.57304148393183])
    water_3 = np.array([-1.0713252576989445, -21.16769176661099, 504.43306896877726])
    water_4 = np.array([-34.40482964759737, 14.19456038206668, 503.3868318328915])
    water_5 = np.array([16.185154425437574, 34.30008903493204, 504.0308151885528])

    # Points in air (for test)
    for p in [p_box_origin_cam,
              p_box_x_axis_cam,
              p_box_y_axis_cam,
              p_water_cam,
              water_1,
              water_2,
              water_3,
              water_4,
              water_5]:
        p_new_object_box = rc.transform_point(p)
        print(f"Transformed point: {p_new_object_box}")

    # Points in water (for test)
    """ tl_water_points = np.array([(840.0, 239.0), (509.0, 153.0)])
    center_water_points = np.array([(1220.0, 630.0), (951.0, 594.0)])
    tr_water_points = np.array([(1614.0, 247.0), (1377.0, 156.0)])
    water_1_points = np.array([(1524.0, 328.0), (1293.0, 250.0)])
    water_2_points = np.array([(1309.0, 446.0), (1049.0, 385.0)])
    water_3_points = np.array([(1540.0, 473.0), (1317.0, 418.0)]) """
    water_1_points = np.array([(986.0, 614.0), (1088.0, 586.0)])
    water_2_points = np.array([(938.0, 900.0), (1036.0, 892.0)])
    water_3_points = np.array([(997.0, 372.0), (1102.0, 328.0)])
    water_4_points = np.array([(681.0, 705.0), (767.0, 683.0)])
    water_5_points = np.array([(1160.0, 895.0), (1273.0, 890.0)])

    for p in [water_1_points,
              water_2_points,
              water_3_points,
              water_4_points,
              water_5_points]:
        underwater_point = rc.correct_underwater_point(p[0], p[1])
        if underwater_point is not None:
            underwater_point = rc.transform_point(underwater_point)
            print(f"Underwater Point: {underwater_point}")

    # Save ray parameters
    # rc.save_parameters("cal_images_250807-1641_10cm/ray_parameters.npz")