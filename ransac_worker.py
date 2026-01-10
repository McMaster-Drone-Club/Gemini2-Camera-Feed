from pyorbbecsdk import *
from math import sqrt
from random import randint

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

        return abs(self.A * x0 + self.B * y0 + self.C * z0 + self.D) / sqrt(self.A ** 2 + self.B ** 2 + self.C ** 2)

    def save_inliers(self, inlier):
        self.inliers.append(inlier)


class RansacWorker:
    def __init__(self, frame_bundle, calibration, image_array, sample_rate=8):
        self.image_array = image_array
        self.depth_matrix = frame_bundle.depth_u16
        self.depth_intrinsics = calibration.depth_intrinsics
        self.extrinsic = calibration.extrinsic
        self.sample_rate = sample_rate
        self.point_mapping = {}

        y_max, x_max, _ = self.image_array.shape
        for u in range(0, x_max, sample_rate):
            for v in range(0, y_max, sample_rate):
                self.point_mapping[(u, v)] = self.convert_to_xyz(u, v) # u, v = x, y

        self.sample_uv = list(self.point_mapping.keys())
        self.sample_uv = [uv for uv in self.sample_uv if uv is not None]

    # returns relative distance data for a pixel u, v
    # index array as y coord, x coord
    # u: x coord v: y coord
    def convert_to_xyz(self, u, v):
        z = self.depth_matrix[v, u]
        if z <= 0:
            return None
        
        return transformation2dto3d(OBPoint2f(u, v), z, self.depth_intrinsics, self.extrinsic)
        
    def run_job(self, thresh=50, n=300, thresh2=0.9):
        y_max, x_max, _ = self.image_array.shape
        best_plane = None

        for _ in range(n):
            i1 = randint(0, len(self.sample_uv) - 1)
            i2 = randint(0, len(self.sample_uv) - 1)
            i3 = randint(0, len(self.sample_uv) - 1)

            p1 = self.sample_uv[i1]
            p2 = self.sample_uv[i2]
            p3 = self.sample_uv[i3]

            p1_xyz = self.point_mapping[p1]
            p2_xyz = self.point_mapping[p2]
            p3_xyz = self.point_mapping[p3]

            if p1_xyz is None or p2_xyz is None or p3_xyz is None:
                continue

            p1_xyz = p1_xyz.x, p1_xyz.y, p1_xyz.z
            p2_xyz = p2_xyz.x, p2_xyz.y, p2_xyz.z
            p3_xyz = p3_xyz.x, p3_xyz.y, p3_xyz.z

            plane = Plane(p1_xyz, p2_xyz, p3_xyz)
            
            if plane.degenerate:
                continue
            
            for u in range(0, x_max, self.sample_rate):
                for v in range(0, y_max, self.sample_rate):
                    point_xyz = self.point_mapping[(u ,v)] # u, v = x, y

                    if point_xyz is None:
                        continue

                    point_xyz = point_xyz.x, point_xyz.y, point_xyz.z
                    distance = plane.distance(point_xyz)

                    if distance is not None and distance < thresh:
                        plane.save_inliers((u, v))

            if best_plane is None or len(plane.inliers) > len(best_plane.inliers):
                best_plane = plane

        return best_plane


        
        
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

