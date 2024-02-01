import numpy as np

class Angles:
    def __init__(self, angle1:int, angle2:int, angle3:int):
        self.angle1 = angle1
        self.angle2 = angle2
        self.angle3 = angle3
    
    def is_valid(self) -> bool:
        return self.angle1 + self.angle2 + self.angle3 == 180
    
    @staticmethod
    def calculate_angle(a:list, b:list, c:list):
        """
            Calculates angle between three points

            Args:
                a (list): The first point.
                b (list): The second point.
                c (list): The third point.

            Returns:
                float: The angle between the three points.
        """
        
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle