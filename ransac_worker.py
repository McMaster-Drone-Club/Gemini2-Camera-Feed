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

        self.inliers = []

        self.degenerate = self.A ** 2 + self.B ** 2 + self.C ** 2 <= 1e-6
  
    
    def distance(self, p1): # returns None ==> collinear
        x0, y0, z0 = p1      

        if self.degenerate:
            return None

        return abs(self.A * x0 + self.B * y0 + self.C * z0 + self.D) / self.normal

    def save_inliers(self, inlier):
        self.inliers.append(inlier)

    def get_hull(self):
        points = np.array(self.inliers, dtype=np.int32).reshape(-1, 1, 2)
        return cv.convexHull(points)
    

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
                p = self.convert_to_xyz((u, v))
                if p is None:
                    continue

                uv.append((u, v))
                xyz.append(p)

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

        
    def run_job(self, job, thresh=50, n=300, thresh2=0.9):
        try:
            y_max, x_max, _ = job.image_array.shape
            best_plane = None

            for _ in range(n):
                i1, i2, i3 = 0, 0, 0

                while i1 == i2 or i2 == i3 or i1 == i3:
                    i1 = randint(0, len(job.uv) - 1)
                    i2 = randint(0, len(job.uv) - 1)
                    i3 = randint(0, len(job.uv) - 1)

                p1 = job.uv[i1]
                p2 = job.uv[i2]
                p3 = job.uv[i3]

                p1_xyz = job.xyz[p1]
                p2_xyz = job.xyz[p2]
                p3_xyz = job.xyz[p3]

                if p1_xyz is None or p2_xyz is None or p3_xyz is None:
                    continue

                p1_xyz = p1_xyz.x, p1_xyz.y, p1_xyz.z
                p2_xyz = p2_xyz.x, p2_xyz.y, p2_xyz.z
                p3_xyz = p3_xyz.x, p3_xyz.y, p3_xyz.z

                plane = Plane(p1_xyz, p2_xyz, p3_xyz)
                
                if plane.degenerate:
                    continue
                
                for u in range(0, x_max, job.sample_rate):
                    for v in range(0, y_max, job.sample_rate):
                        point_xyz = job.xyz[(u, v)]  # u, v = x, y

                        if point_xyz is None:
                            continue

                        point_xyz = point_xyz.x, point_xyz.y, point_xyz.z
                        distance = plane.distance(point_xyz)

                        if distance is not None and distance < thresh:
                            plane.save_inliers((u, v))

                if best_plane is None or len(plane.inliers) > len(best_plane.inliers):
                    best_plane = plane

            if best_plane is None or len(best_plane.inliers) < 3:
                self.state.clear_wall()
                return None
            
            self.state.update_wall(best_plane.get_hull())
            return best_plane
        
        except Exception as e:
            print("Failed to run RANSAC")
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

