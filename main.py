# 1.Prerequisites
# Make sure you have already installed the pyorbbecsdk library in your local Python environment using the steps in Installation above.

from pyorbbecsdk import *
# 2.create pipeline 
pipeline = Pipeline()

# 3.start pipeline
pipeline.start()

# 4.wait for frames
frames = pipeline.wait_for_frames(100)

# 5.get frames
color_frame = frames.get_color_frame()
depth_frame = frames.get_depth_frame()

# 6.Render frames using OpenCV

# 7.stop pipeline
pipeline.stop()