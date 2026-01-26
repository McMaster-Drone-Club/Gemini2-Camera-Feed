from pyorbbecsdk import *
from random import randint
import cv2 as cv
import numpy as np
from threading import Thread

class Plane:
    # Ax + By + Cz + d = 0
    def __init__(self, p1, p2, p3): # all 3d tuples
        a = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
        b = p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]

        self.n = np.array([a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]], dtype=float)
        
        self.normal = np.linalg.norm(self.n)
        
        self.A = self.n[0]
        self.B = self.n[1]
        self.C = self.n[2]
        self.D = -(self.A * p1[0] + self.B * p1[1] + self.C * p1[2]) 

        self.inliers_uv = []
        self.inliers_xyz = []


    def is_ground(self, gravity, thresh=0.8): # closer to 0 ==> more perpendicular to gravity
        g_hat = gravity / np.linalg.norm(gravity)
        n_hat = self.n / self.normal

        return abs(np.dot(n_hat, g_hat)) >= thresh
        
    
    def distance(self, p1): # returns None ==> collinear; p1 in the form (X, Y, Z, 1)
        p2 = [self.A, self.B, self.C]
        return np.abs((np.dot(p1, p2) + self.D) / self.normal)

    def get_hull(self):
        if len(self.inliers_uv) < 3:
            return None

        points = np.array(self.inliers_uv, dtype=np.int32).reshape(-1, 1, 2)
        return cv.convexHull(points)


class Wall:
    def __init__(self, hull, normal):
        self.hull = hull
        self.normal = normal
    

class RansacJob:
    def __init__(self, frame_bundle, calibration, image_array, sample_rate=8):
        self.frame_bundle = frame_bundle
        self.calibration = calibration
        self.image_array = image_array
        self.depth_matrix = frame_bundle.depth_u16
        self.depth_intrinsics = calibration.depth_intrinsics
        self.extrinsic = calibration.extrinsic
        self.sample_rate = sample_rate
        
        y_max, x_max, _ = self.image_array.shape
        uv = []
        xyz = []

        for u in range(0, x_max, sample_rate):
            for v in range(0, y_max, sample_rate):
                p = self.convert_to_xyz(u, v)
                if p is None:
                    continue

                uv.append((u, v))
                xyz.append((p.x, p.y, p.z))

        self.uv = np.array(uv, dtype=np.int32)
        self.xyz = np.array(xyz, dtype=np.float32)


    # returns relative distance data for a pixel u, v
    # index array as y coord, x coord
    # u: x coord v: y coord
    def convert_to_xyz(self, u, v):
        z = self.depth_matrix[v, u]
        if z <= 0:
            return None
        
        return transformation2dto3d(OBPoint2f(u, v), z, self.depth_intrinsics, self.extrinsic)


class RansacWorker:
    def __init__(self, state, mav):
        self.state = state
        self.mav = mav

        self.g = np.array([0.0, 0.0, -1.0]) # representation of gravity in 3D world space, needs to be transformed to camera space
        self.camera_tilt = np.array([
            [np.cos(np.pi / 4), 0, np.sin(np.pi / 4)],
            [0, 1, 0],
            [-np.sin(np.pi / 4), 0, np.cos(np.pi / 4)]
        ], dtype=float)

    def submit_job(self, job):
        if self.state.is_ransac_busy():
            return False
        
        self.state.set_ransac_busy(True)
        Thread(target=self.run_job, args=(job, 50, 300, 0.9), daemon=True).start()
        return True

    
    def run_job(self, job, thresh=50, n=300, thresh2=0.9):
        # get pitch, roll, yaw from IMU on FC
        # measure the camera tilt
        # check the imu pitch roll yaw return convention
        snap = self.state.snapshot()
        pitch = snap["pitch"]
        roll = snap["roll"]

        pitch_transform = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ], dtype=float)

        roll_transform = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ], dtype=float)

        # TODO: Check camera axis direction to determine order of pitch and roll
        gravity = self.camera_tilt @ (roll_transform @ pitch_transform @ self.g) # world -> body -> camera

        try:
            best_plane = None

            for _ in range(n):
                i1, i2, i3 = 0, 0, 0

                while i1 == i2 or i2 == i3 or i1 == i3:
                    i1 = randint(0, len(job.uv) - 1)
                    i2 = randint(0, len(job.uv) - 1)
                    i3 = randint(0, len(job.uv) - 1)

                p1_xyz = job.xyz[i1]
                p2_xyz = job.xyz[i2]
                p3_xyz = job.xyz[i3]

                plane = Plane(p1_xyz, p2_xyz, p3_xyz)
                
                if plane.A ** 2 + plane.B ** 2 + plane.C ** 2 <= 1e-6:
                    continue
                
                # array of distances from each xyz coordinate to plane
                distances = plane.distance((job.xyz[:, 0], job.xyz[:, 1], job.xyz[:, 2]))
                # filter for distances
                mask = distances < thresh
                # indexing just the uv's and xyz's that pass the filter
                plane.inliers_uv = job.uv[mask]
                plane.inliers_xyz = job.xyz[mask]

                # if plane.is_ground(self.gravity):
                #     continue

                if best_plane is None or len(plane.inliers_uv) > len(best_plane.inliers_uv):
                    best_plane = plane

            if best_plane is None or len(best_plane.inliers_uv) < 3:
                self.state.clear_wall()
                return None

            wall = Wall(best_plane.get_hull(), best_plane.n / best_plane.normal)
            self.state.update_wall(wall) # currently the wall is just a hull, fix this to include the normal too
            
            return best_plane
        
        except Exception as e:
            print("Failed to run RANSAC " + repr(e))
            self.state.clear_wall()
            return None
        
        finally:
            self.state.set_ransac_busy(False)

        
        

"""

RANSAC loop:

randomly pick 3 points

compute the plane from them

compute distance of every point to that plane

count points with distance < threshold (these are inliers)

keep the plane with the most inliers

optionally refit using all inliers (more accurate)

"""

