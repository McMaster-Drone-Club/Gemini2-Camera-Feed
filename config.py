class AppConfig():
    def __init__(self, model_path="yolov10m.pt", mask_code="rgbpgoy"):
        self.min_depth = 20  # 20mm
        self.max_depth = 10000  # 10000mm = 10 m

        self.yolo_interval = 20
        
        self.window_width = 1280
        self.window_height = 720

        self.light_blue = (95, 80, 50)
        self.dark_blue  = (130, 255, 255)

        self.light_purple = (125, 40, 50)
        self.dark_purple = (155, 255, 255)

        self.light_green = (40, 50, 50)
        self.dark_green = (75, 255, 255)

        self.light_red1 = (0, 80, 50)
        self.dark_red1  = (10, 255, 255)

        self.light_red2 = (170, 80, 50)
        self.dark_red2  = (179, 255, 255)

        self.light_orange = (15, 80, 50)
        self.dark_orange = (25, 255, 255)

        self.light_yellow = (20, 40, 60)
        self.dark_yellow = (35, 255, 255)
        
        self.model_path = model_path 
        self.mask_code = mask_code