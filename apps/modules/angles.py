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
        # Convert points into numpy vectors
        point1 = np.array(a)
        point2 = np.array(b)
        point3 = np.array(c)

        # Calculate the vectors between the points
        vector1 = point1 - point2
        vector2 = point3 - point2

        # Use the cosine law to calculate the angle
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(cosine_angle)
        
        # Convert the angle to degrees
        angle_degrees = np.degrees(angle)

        return angle_degrees