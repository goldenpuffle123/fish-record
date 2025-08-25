from ray_tracing_projection import RefractionCalibration
import numpy as np
import time

rc = RefractionCalibration("cal_images_250807-1641_10cm/stereo_matrices.npz", upload_path="cal_images_250807-1641_10cm/ray_parameters.npz")

parts=6
start_time = time.perf_counter()
for i in range(10000):
    for j in range(parts):
        new_points = np.array([(986.0, 614.0), (1088.0, 586.0)])
        underwater_point = rc.correct_underwater_point(new_points[0], new_points[1])
        if underwater_point is not None:
            underwater_point = rc.transform_point(underwater_point)
print(f"Ran for {time.perf_counter() - start_time} seconds")