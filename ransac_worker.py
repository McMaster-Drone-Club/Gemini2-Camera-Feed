from pyorbbecsdk import *
from math import sqrt
from random import randint
import cv2 as cv
import numpy as np
from threading import Thread

class Plane:
    # Ax + By + Cz + d = 0
    def __init__(self, p1, p2, p3): # all 3d tuples
        a = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
        b = p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]

        self.n = [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]
        
        self.normal = sqrt(self.n[0] ** 2 + self.n[1] ** 2 + self.n[2] ** 2)
        
        self.A = self.n[0]
        self.B = self.n[1]
        self.C = self.n[2]
        self.D = -(self.A * p1[0] + self.B * p1[1] + self.C * p1[2]) 

        self.inliers_uv = []
        self.inliers_xyz = []

      
    
    def distance(self, p1): # returns None ==> collinear
        x0, y0, z0 = p1      

        return np.abs(self.A * x0 + self.B * y0 + self.C * z0 + self.D) / self.normal

    def get_hull(self):
        if len(self.inliers_uv) < 3:
            return None

        points = np.array(self.inliers_uv, dtype=np.int32).reshape(-1, 1, 2)
        return cv.convexHull(points)
    
    def is_ground(self, pitch=0.0, roll=0.0, threshold=0.85):
        """
        Accounts for drone pitch and roll.
        
        Uses dot product: if |n·g| ≈ 1, plane is horizontal (ground).
                          if |n·g| ≈ 0, plane is vertical (wall).
        
        Args:
            pitch: Drone pitch angle in radians (rotation around X-axis/right direction).
                   Positive = nose up. Default 0 (level).
            roll: Drone roll angle in radians (rotation around Z-axis/forward direction).
                  Positive = right wing down. Default 0 (level).
            threshold: Dot product threshold (0-1). Higher = more strict in identifying ground.
                      0.85 means the plane normal must be within ~32 degrees of vertical.
        
        Returns:
            True if plane is likely the ground, False if it's likely a wall.
        """
        if self.normal == 0:
            return False
        
        # Start with gravity pointing down in camera frame: [0, 1, 0]
        gravity_vec = np.array([0.0, 1.0, 0.0])
        
        # Apply drone pitch (rotation around X-axis)
        # NOTE: When drone pitches up, ground plane normal shifts
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        pitch_transform = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cos_pitch, -sin_pitch],
            [0.0, sin_pitch, cos_pitch]
        ])
        
        # drone roll -> ground plane normal shifts sideways
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        roll_transform = np.array([
            [cos_roll, 0.0, sin_roll],
            [0.0, 1.0, 0.0],
            [-sin_roll, 0.0, cos_roll]
        ])
        
        # roll first, then pitch (linear algebra works out this way)
        combined_rotation = pitch_transform @ roll_transform
        gravity_vec = combined_rotation @ gravity_vec
        
        normalized_normal = np.array([self.A, self.B, self.C]) / self.normal
        
        #  magntiude of dot product (up or down doesn't matter here)
        dot_product = np.abs(np.dot(normalized_normal, gravity_vec))
        
        # If dp ~ 1, the plane is parallel to effective gravity (ground)
        # If dp ~ 0, the plane is perpendicular to gravity (wall)
        return dot_product > threshold
    

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
    def __init__(self, state):
        self.state = state
        
    
    def submit_job(self, job):
        if self.state.is_ransac_busy():
            return False
        
        self.state.set_ransac_busy(True)
        Thread(target=self.run_job, args=(job, 50, 300, 0.9), daemon=True).start()
        return True

        
    def run_job(self, job, thresh=50, n=300, thresh2=0.9, ground_threshold=0.85, pitch=0.0, roll=0.0):
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
                
                # Skip ground planes - we only want to detect walls
                # Pass drone pitch and roll to account for tilted drone
                if plane.is_ground(pitch=pitch, roll=roll, threshold=ground_threshold):
                    continue
                
                # array of distances from each xyz coordinate to plane
                distances = plane.distance((job.xyz[:, 0], job.xyz[:, 1], job.xyz[:, 2]))
                # filter for distances
                mask = distances < thresh
                # indexing just the uv's and xyz's that pass the filter
                plane.inliers_uv = job.uv[mask]
                plane.inliers_xyz = job.xyz[mask]

                if best_plane is None or len(plane.inliers_uv) > len(best_plane.inliers_uv):
                    best_plane = plane

            if best_plane is None or len(best_plane.inliers_uv) < 3:
                self.state.clear_wall()
                return None
            
            self.state.update_wall(best_plane.get_hull())
            return best_plane
        
        except Exception as e:
            print("Failed to run RANSAC " + repr(e))
            self.state.clear_wall()
            return None
        
        finally:
            self.state.set_ransac_busy(False)

        
        
        # pick 3 points from image array
        # convert them to 3d coordintes
        # compute the plane
        # compute distance
        #count points with distance < threshold (these are inliers)
        #keep the plane with the most inliers


"""

RANSAC loop:

randomly pick 3 points

compute the plane from them

compute distance of every point to that plane

count points with distance < threshold (these are inliers)

keep the plane with the most inliers

optionally refit using all inliers (more accurate)

"""

"""

to filter out ground and other 'horizontal' surfaces, check if |n * g| is close to 1 because 
then the plane's normal and gravity vector will be close to perpendicular

apply a rotation matrix to account for the drone's pitch and yaw
"""