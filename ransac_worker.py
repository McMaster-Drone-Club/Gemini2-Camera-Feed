from pyorbbecsdk import *
from math import sqrt,sin,cos,acos,degrees,radians
from random import randint
import cv2 as cv
import numpy as np
from threading import Thread
import unittest


class PlaneClassifier:
    @staticmethod
    def transform_normal(plane_normal, pitch_rad, roll_rad, cam_tilt_rad=0.0):
        """
        Transforms camera plane normal [A, B, C] into gravity-aligned world frame.
        """
        normal_vector = np.array(plane_normal, dtype=np.float64)
        norm = np.linalg.norm(normal_vector)
        if norm < 1e-6:
            return None
        normal_vector /= norm

        #Map Camera Frame to Drone Frame 
        # X_drone = Z_camera, Y_Drone = X_Camera, Z_Drone = Y_Camera
        normal_vector = np.array([normal_vector[2], normal_vector[0], -normal_vector[1]])


        #Apply fixed camera mounting pitch angle if present
        if abs(cam_tilt_rad) > 1e-5:
            R_tilt = np.array([
                [cos(cam_tilt_rad), 0, sin(cam_tilt_rad)],
                [0, 1, 0],
                [-sin(cam_tilt_rad), 0, cos(cam_tilt_rad)]
            ])
            normal_vector = R_tilt @ normal_vector
        #Apply IMU Pitch and Roll
        Rx = np.array([
            [1, 0, 0],
            [0, cos(roll_rad), -sin(roll_rad)],
            [0, sin(roll_rad), cos(roll_rad)]
        ])
        Ry = np.array([
            [cos(pitch_rad), 0, sin(pitch_rad)],
            [0, 1, 0],
            [-sin(pitch_rad), 0, cos(pitch_rad)]
        ])

        return Ry @ (Rx @ normal_vector)
    
    @staticmethod
    def classify(world_normal, floor_thresh_deg=15.0, wall_thresh_deg=15.0):
        """
        Classifies plane normal into wall, floor, ambiguous, or invalid.
        """
        if world_normal is None:
            return "invalid"

        # Abs dot product with gravity vector [0, 0, 1]
        cos_angle = abs(world_normal[2])
        cos_angle = np.clip(cos_angle, 0.0, 1.0)
        angle_from_vertical = degrees(acos(cos_angle))

        if angle_from_vertical <= floor_thresh_deg:
            return "floor"
        elif angle_from_vertical >= (90.0 - wall_thresh_deg):
            return "wall"
        else:
            return "ambiguous"


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

        self.label = "unclassified"
        self.confidence = 0.0
      
    def distance(self, p1): # returns None ==> collinear
        x0, y0, z0 = p1      

        return np.abs(self.A * x0 + self.B * y0 + self.C * z0 + self.D) / self.normal

    def get_hull(self):
        if len(self.inliers_uv) < 3:
            return None

        points = np.array(self.inliers_uv, dtype=np.int32).reshape(-1, 1, 2)
        return cv.convexHull(points)

    def classify_plane(self, pitch_rad, roll_rad, cam_tilt_rad=0.0):
        world_normal = PlaneClassifier.transform_normal([self.A, self.B, self.C], pitch_rad, roll_rad, cam_tilt_rad)
        self.label = PlaneClassifier.classify(world_normal)
        return self.label  

    def compute_metrics(self, total_points, distances_array, thresh):
        #Detemrines confidence and root mean square error
        self.confidence = len(self.inliers_xyz) / total_points if total_points > 0 else 0.0
        inlier_dists = distances_array[distances_array < thresh]
        self.rmse = float(np.sqrt(np.mean(inlier_dists**2))) if len(inlier_dists) > 0 else float('inf')  

    
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

        
    def run_job(self, job, thresh=50, n=300, thresh2=0.9):
        try:
            best_plane = None
            best_distances = None

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

                if best_plane is None or len(plane.inliers_uv) > len(best_plane.inliers_uv):
                    best_plane = plane
                    best_distances = distances
        
            if best_plane is None:
                self.state.clear_wall()
                return None

            #Retrieves attitude from shared state
            pitch_rad,roll_rad,yaw_rad = self.state.get_attitude_angles()

            #Classifies plane and determines confidence
            label = best_plane.classify_plane(pitch_rad, roll_rad)
            best_plane.compute_metrics(len(job.xyz), best_distances, thresh)

            if label in ["ambiguous", "invalid"] or best_plane.confidence < 0.15 or best_plane.rmse > 30.0:
                self.state.clear_wall()
                return None
    
            self.state.update_wall(best_plane.get_hull(), label) 
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
        #keep the plane with the most inliers if wall with >90% certainty

class TestPlaneClassifier(unittest.TestCase):

    def test_level_floor_classification(self):
        # Level hover over floor; surface normal points up (-Y in cam frame)
        cam_n = [0, -1, 0]
        world_n = PlaneClassifier.transform_normal(cam_n, pitch_rad=0, roll_rad=0)
        self.assertEqual(PlaneClassifier.classify(world_n), "floor")

    def test_level_front_wall_classification(self):
        # Level hover facing front wall (-Z in cam frame)
        cam_n = [0, 0, -1]
        world_n = PlaneClassifier.transform_normal(cam_n, pitch_rad=0, roll_rad=0)
        self.assertEqual(PlaneClassifier.classify(world_n), "wall")

    def test_level_side_wall_classification(self):
        # Level hover facing side wall (-X in cam frame)
        cam_n = [-1, 0, 0]
        world_n = PlaneClassifier.transform_normal(cam_n, pitch_rad=0, roll_rad=0)
        self.assertEqual(PlaneClassifier.classify(world_n), "wall")

    def test_pitched_drone_front_wall(self):
        # Drone pitched down 20° facing front wall
        pitch = radians(-20)
        cam_n = [0, -sin(radians(20)), -cos(radians(20))]
        world_n = PlaneClassifier.transform_normal(cam_n, pitch_rad=pitch, roll_rad=0)
        self.assertEqual(PlaneClassifier.classify(world_n), "wall")

    def test_rolled_drone_floor(self):
        # Drone rolled right(CW) 15° over floor
        roll = -radians(15)
        cam_n = [-sin(radians(15)), -cos(radians(15)), 0]
        world_n = PlaneClassifier.transform_normal(cam_n, pitch_rad=0, roll_rad=roll)
        self.assertEqual(PlaneClassifier.classify(world_n), "floor")

    def test_fixed_camera_mount_tilt(self):
        # Level drone with camera mechanically tilted down 15°
        cam_tilt = radians(15)
        cam_n = [0, -cos(cam_tilt), -sin(cam_tilt)]
        world_n = PlaneClassifier.transform_normal(cam_n, pitch_rad=0, roll_rad=0, cam_tilt_rad=cam_tilt)
        self.assertEqual(PlaneClassifier.classify(world_n), "floor")

    def test_ambiguous_sloped_ramp(self):
        # Level hover facing a 45° sloped surface
        cam_n = [0, -cos(radians(45)), -sin(radians(45))]
        world_n = PlaneClassifier.transform_normal(cam_n, pitch_rad=0, roll_rad=0)
        self.assertEqual(PlaneClassifier.classify(world_n), "ambiguous")

    def test_invalid_zero_normal(self):
        # zero normal input
        cam_n = [0, 0, 0]
        world_n = PlaneClassifier.transform_normal(cam_n, pitch_rad=0, roll_rad=0)
        self.assertEqual(PlaneClassifier.classify(world_n), "invalid")

if __name__ == "__main__":
    unittest.main()


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