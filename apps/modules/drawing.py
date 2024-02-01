import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tempfile
import subprocess
try:
    from angles import Angles
except:
    from .angles import Angles

class Drawing:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils
        self.custom_pose_connections = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
        ]

    @staticmethod
    def initialize_camera(camera) -> int:
        """
            Function to Shutdown or Setup the camera.
            Returns:
                int: 0 if function run correctly
        """
        # Release the camera if it's open
        if camera is not None and camera.isOpened():
            camera.release()

        # Re-initialize the camera
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError("Could not initialize camera")

        # Indicate that the camera is now active
        return True, camera
    
    @staticmethod
    def coordonnate_association(landmarks, mp_pose):
        """ 
            Function to associate prediction with body part
            Args:
                landmarks: list of landmarks
                mp_pose: mediapipe pose class 
            Returns:
                list of landmarks (shoulder, elbow, wrist, knee, ankle, hip, horizontal)
        """
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        horizontal = [hip[0] - 1, hip[1]]

        return shoulder, elbow, wrist, knee, ankle, hip, horizontal
    
    @staticmethod
    def draw_arc(frame: np.ndarray, center: tuple, start_point: tuple, end_point: tuple, color: tuple, transparency: float = 0.5, is_back_angle:bool=False) -> np.ndarray:
        """
            Draw arc corresponding to the angle.
            Args:
                frame (numpy.ndarray): The frame to draw on.
                center (tuple): The center of the arc.
                start_point (tuple): The start point of the arc.
                end_point (tuple): The end point of the arc.
                color (tuple): The color of the arc.
                transparency (float): The transparency of the arc.
                is_back_angle (bool): Whether the angle is back angle or not. Default is False.
            Returns:
                None
        """
        overlay = frame.copy()
        # Define vector parameters
        vector_color=(0, 0, 255)
        vector_scale=0.5
        dot_radius=3
        # Convert points into numpy vectors
        center_np = np.array(center)
        start_point_np = np.array(start_point)
        end_point_np = np.array(end_point)
        # Calculate the vectors
        vec_start = start_point_np - center_np
        vec_end = end_point_np - center_np
        # Scale down the vectors
        vec_start = vec_start * vector_scale
        vec_end = vec_end * vector_scale
        # Calculate new start and end points for the shorter vectors
        start_point_short = center_np + vec_start
        end_point_short = center_np + vec_end
        # Calculate axes length as the longest distance to the center
        axes_length = (int(np.linalg.norm(vec_start) / 2), int(np.linalg.norm(vec_end) / 2))
        if is_back_angle:
            vec_end = end_point_np - center_np  # Hip-acromion vector
            # Use the length of the hip-acromion vector for both axes to avoid distortion
            axes_length = (int(np.linalg.norm(vec_end) / 2), int(np.linalg.norm(vec_end) / 2))
            
            angle_start = 180  # Horizontal vector will start at 0 degrees
            angle_hip_acromion  = np.degrees(np.arctan2(vec_end[1], vec_end[0])) # Calculate angle of hip-acromion vector
            angle_end = (angle_hip_acromion + 180) % 360

            if angle_end < angle_start:
                angle_end += 180
            angle_end %= 360
        else:
            angle_start = np.degrees(np.arctan2(vec_start[1], vec_start[0]))
            angle_end = np.degrees(np.arctan2(vec_end[1], vec_end[0]))
        cv2.ellipse(overlay, tuple(center), axes_length, 0, angle_start, angle_end, color, thickness=-1)
        cv2.line(overlay, tuple(center), tuple(start_point_short.astype(int)), vector_color, thickness=2)
        cv2.line(overlay, tuple(center), tuple(end_point_short.astype(int)), vector_color, thickness=2)
        cv2.circle(overlay, tuple(start_point_short.astype(int)), dot_radius, vector_color, thickness=-1)
        cv2.circle(overlay, tuple(end_point_short.astype(int)), dot_radius, vector_color, thickness=-1)
        cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0, frame)

        return frame

    def compute_and_drawing_processing (self, frame, i:int=0, enable_cvtColor:bool=True):
        """
            Function to compute and drawing processing
            Args:
                frame: image to process
                i: index of frame
            Returns:
                img: image with processing
                angles_value: dict of angles (bras, jambe, bras/buste, tronc, dos)
        """
        try:
            # Change color of frame for better processing
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            results = self.pose.process(frame)
            frame.flags.writeable = True
            angles_value = {}

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                shoulder, elbow, wrist, knee, ankle, hip, horizontal = Drawing.coordonnate_association(landmarks, self.mp_pose)

                for connection in self.custom_pose_connections:
                    start_landmark = connection[0]
                    end_landmark = connection[1]
                    start_point = (int(landmarks[start_landmark].x * w), int(landmarks[start_landmark].y * h))
                    end_point = (int(landmarks[end_landmark].x * w), int(landmarks[end_landmark].y * h))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 3) 

                hip_pixel = (int(hip[0] * w), int(hip[1] * h))
                line_length = 80
                horizontal_start = (hip_pixel[0] - line_length, hip_pixel[1])
                horizontal_end = hip_pixel
                cv2.line(frame, horizontal_start, horizontal_end, (255, 0, 0), 1)

                shoulder = (int(shoulder[0] * w), int(shoulder[1] * h))
                elbow = (int(elbow[0] * w), int(elbow[1] * h))
                wrist = (int(wrist[0] * w), int(wrist[1] * h))
                knee = (int(knee[0] * w), int(knee[1] * h))
                ankle = (int(ankle[0] * w), int(ankle[1] * h))
                hip = (int(hip[0] * w), int(hip[1] * h))
                horizontal = (int(horizontal[0] * w), int(horizontal[1] * h))

                # Dessinez les ellipses pour chaque angle calculé
                self.draw_arc(frame, elbow, wrist, shoulder, color=(255, 0, 0, 128), transparency=0.5)
                self.draw_arc(frame, knee, hip, ankle, color=(0, 255, 0), transparency=0.5)
                self.draw_arc(frame, shoulder, hip, wrist, color=(0, 0, 255), transparency=0.5)
                self.draw_arc(frame, hip, ankle, shoulder, color=(255, 255, 0), transparency=0.5, is_back_angle=True)

                angles_value = {
                    "i": i,
                    "bras": round(Angles.calculate_angle(shoulder, elbow, wrist), 2),
                    "jambe": round(Angles.calculate_angle(hip, knee, ankle), 2),
                    "Bras/buste": round(Angles.calculate_angle(hip, shoulder, wrist), 2),
                    "tronc": round(Angles.calculate_angle(shoulder, hip, ankle), 2),
                    "dos": round(Angles.calculate_angle(horizontal, hip, shoulder), 2)
                }

                # Reset origin color for display
                if enable_cvtColor:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        except Exception as e:
            print("[ERROR] Unable to process the image: ", e)
            pass

        return frame, angles_value

    def process_livestream(self, is_camera_active, camera):
        """
            Function to display computer vision solution 
            and draw the skeleton on the screen.
            Returns:
                None
        """
        while is_camera_active:
            # read the camera frame
            success, frame = camera.read()
            if not success:
                # If the frame could not be read, release the camera and exit the loop.
                camera.release()
                break
                #raise RuntimeError("Failed to read camera frame")

            frame = cv2.flip(frame, 1)
            
            frame, angles_value = self.compute_and_drawing_processing(frame=frame)

            # Initialize frame_bytes as an empty byte string
            frame_bytes = b''

            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                frame_bytes = buffer.tobytes()

            # Only yield if frame_bytes is not empty
            if frame_bytes:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )
            else:
                # Handle the case where encoding failed, e.g., by logging an error or yielding a placeholder image
                print("[ERROR] Encoding failed")
                pass

            # Update UI with the processed frame
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Ensure frame is in BGR format for encoding
            frame_bytes = buffer.tobytes()

        # Release the camera and close the window if is_camera_active == False
        camera.release()

    def process_frame_file(self, image_path:str):
        """
            Calculate and draw angle in Image
            Args:
                image_path (str): The path to the image.
            Returns:
                tuple: The image with angles drawn and the angles data.
        """
        try:
            # Lecture de l'image
            frame = cv2.imread(image_path)

            # Processing et drawing sur la frame
            frame, angles_value = self.compute_and_drawing_processing(frame=frame, enable_cvtColor=False)

            # Convertir l'frame CV2 en un objet frame PIL pour l'affichage dans Streamlit
            pil_image = Image.fromarray(frame)
            
            return pil_image, angles_value
        
        except Exception as e:
            print("[ERROR] Unable to process the image: ", e)

    def process_video_file(self, video_path:str) -> tuple[str, list, int]:
        """
            Calculate and draw angle in Video
            Args:
                video_path (str): The path to the video.
            Returns:
                tuple: 
                    temp_file_out.name (str): The video filepath with angles drawn
                    angles_values (list): Angles data with frame code
                    fps_input (int): FPS of the input video
        """
        vid = cv2.VideoCapture(video_path)
        temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".mp4")
        temp_file_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))

        codec = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_file.name, codec, fps_input, (width, height))

        i = 0
        angles_values = []
        while True:
            try:
                ret, frame = vid.read()
                if not ret:
                    break

                frame, angles_value = self.compute_and_drawing_processing(frame, i)

                angles_values.append(angles_value)
                i+= 1
                out.write(frame)
            except Exception as e:
                print("[ERROR] Unable to process the video: ", e)

        # Fermeture des flux video
        vid.release()
        out.release()

        # Convertion de la suite de frame (out) en format exploitable par les navigateurs web
        subprocess.call(args=f"ffmpeg -y -i {temp_file.name} -c:v libx264 {temp_file_out.name}".split(" "))            
            
        return temp_file_out.name, angles_values, fps_input
    
    