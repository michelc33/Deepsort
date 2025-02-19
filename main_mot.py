import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from typing import List, Tuple, Dict
from collections import deque  # For EMA gallery implementation

# --- Helper Functions ---

def download_reid_resnet():
    """
    Downloads the pretrained person ReID ResNet model.  Replace with the
    appropriate download logic for your specific model storage/hosting.
    """
    model_path = "personReIDResNet_v2.pth"  # Changed to .pth extension for PyTorch

    # Check if the model exists
    if not os.path.exists(model_path):
        print("Downloading Pretrained Person ReID Network (~198 MB)")
        # Replace with actual download code (e.g., using urllib.request.urlretrieve)
        # This is a placeholder for a real download.
        print(f"Placeholder: Downloading model to {model_path}")

        # After downloading, you'll need to load the model.  The rest of the code assumes it's available.
        # Example of creating a dummy model (replace with loading your actual model)
        #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        #torch.save(model, model_path)
        print("Download complete (placeholder implementation).  Ensure you have your ReID model loaded.")
    else:
        print("Pretrained Person ReID Network already exists.")


def run_reid_net(net, frame, detections):
    """
    Extracts appearance features from detections using the ReID network.

    Args:
        net: The ReID network (PyTorch model).
        frame: The current video frame (NumPy array).
        detections: A list of detection dictionaries, where each dictionary contains
                    at least 'bbox' (bounding box in [x, y, width, height] format)
                    and potentially other attributes like 'score'.  The function adds
                    the 'appearance' key to each detection.

    Returns:
        The updated list of detection dictionaries, with 'appearance' added.
    """
    if not detections:
        return []

    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert NumPy array to PIL Image
        transforms.Resize((128, 64)), # Or whatever input size your ReID expects
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
    ])

    for detection in detections:
        bbox = detection['bbox']
        x, y, w, h = map(int, bbox)  # Ensure integer coordinates

        # Error handling for bounding boxes outside the image
        x1, y1, x2, y2 = x, y, x + w, y + h
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > frame.shape[1]: x2 = frame.shape[1]
        if y2 > frame.shape[0]: y2 = frame.shape[0]

        cropped_person = frame[y1:y2, x1:x2]

        # Handle cases where the crop is empty (e.g., bbox too small or outside image).  Critical to avoid crashes.
        if cropped_person.size == 0:
            print(f"Warning: Empty crop for detection {detection}.  Setting default appearance vector.")
            # Create a dummy feature vector (e.g., zeros)
            appearance_vect = torch.zeros(2048) # Adjust dimension to your ReID network's output
        else:
            # Convert to RGB if necessary and handle grayscale
            if len(cropped_person.shape) == 2:  # Grayscale image
                cropped_person = cv2.cvtColor(cropped_person, cv2.COLOR_GRAY2RGB) # Convert to RGB

            cropped_person_tensor = transform(cropped_person).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():  # Disable gradient calculation during inference
                appearance_vect = net(cropped_person_tensor).squeeze() # Remove batch dimension

            # Normalize the feature vector (important for cosine distance)
            appearance_vect = appearance_vect / torch.linalg.norm(appearance_vect)

        detection['appearance'] = appearance_vect

    return detections


def delete_out_of_frame_tracks(tracker, confirmed_tracks, frame_width, frame_height):
    """
    Deletes tracks whose bounding boxes are entirely out of the video frame.

    Args:
        tracker: The tracker object (implementation depends on your chosen tracking library).
        confirmed_tracks: A list of track objects (structure depends on your chosen tracking library).
        frame_width: The width of the video frame.
        frame_height: The height of the video frame.
    """

    tracks_to_delete = []  # Collect track IDs to delete

    for track in confirmed_tracks:
        bbox = track['bbox'] # Assumes track has bbox attribute
        x, y, w, h = bbox

        if x + w < 0 or x > frame_width or y + h < 0 or y > frame_height:
            tracks_to_delete.append(track['id'])  # Assumes track has an 'id' attribute

    for track_id in tracks_to_delete:
        tracker.delete_track(track_id)  # Delete the track from the tracker
    return tracker


def iou(bbox1, bbox2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: Bounding box 1 in [x, y, width, height] format.
        bbox2: Bounding box 2 in [x, y, width, height] format.

    Returns:
        The IoU value (a float between 0 and 1).
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate coordinates of the intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of each bounding box
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    # Calculate the area of union
    union_area = bbox1_area + bbox2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def mahalanobis_distance(track, detection, measurement_noise):
    """
    Calculates the Mahalanobis distance between a track and a detection.

    Args:
        track: The track object (assumed to have 'state' and 'state_covariance' attributes).
               'state' is assumed to be in [x, y, width, height, vx, vy, vw, vh] format.
        detection: The detection dictionary with 'bbox' key.
        measurement_noise: The measurement noise covariance matrix (NumPy array).

    Returns:
        The Mahalanobis distance (a float).
    """
    track_state = track['state'][:4]  # x, y, width, height (predicted measurement)
    track_covariance = track['state_covariance'][:4, :4]

    detection_measurement = detection['bbox']  # x, y, width, height

    innovation = np.array(detection_measurement) - track_state
    innovation_covariance = track_covariance + measurement_noise

    try:
        distance = innovation @ np.linalg.inv(innovation_covariance) @ innovation
    except np.linalg.LinAlgError:
        # Handle singular matrix case (e.g., by returning a large distance)
        return float('inf')

    return distance


def cosine_distance(track, detection, appearance_momentum=0.0):
    """
    Calculates the minimum cosine distance between a detection's appearance feature and
    the track's appearance history (gallery or EMA).

    Args:
        track: The track object (assumed to have an 'appearance_history' attribute).
               If EMA is used, 'appearance_history' is assumed to be a single vector.
        detection: The detection dictionary with 'appearance' key.
        appearance_momentum (float): momentum value when using Exponential Moving Average
    Returns:
        The minimum cosine distance (a float).
    """
    detection_appearance = detection['appearance']

    if 'appearance_history' in track:

        if isinstance(track['appearance_history'], deque): #Gallery

            min_distance = float('inf')
            for appearance in track['appearance_history']:
                distance = 1 - torch.dot(detection_appearance, appearance)
                min_distance = min(min_distance, distance.item())
            return min_distance

        else: #EMA

            distance = 1 - torch.dot(detection_appearance, track['appearance_history'])
            return distance.item()

    else: #EMA Gallery

        min_distance = float('inf')
        for appearance in track['appearance_history']:
            distance = 1 - torch.dot(detection_appearance, appearance)
            min_distance = min(min_distance, distance.item())
        return min_distance



# --- DeepSORT Tracker Class ---
import os
class DeepSORT:
    """
    Implements the DeepSORT multi-object tracking algorithm.
    """

    def __init__(self,
                 confirmation_threshold: Tuple[int, int] = (2, 2),
                 deletion_threshold: Tuple[int, int] = (5, 5),
                 appearance_update: str = "Gallery",  # "Gallery", "EMA", "EMA Gallery"
                 max_num_appearance_frames: int = 50,
                 mahalanobis_assignment_threshold: float = 10.0,
                 appearance_assignment_threshold: float = 0.4,
                 iou_assignment_threshold: float = 0.95,
                 appearance_weight: float = 0.02,
                 frame_size: Tuple[int, int] = (1288, 964),
                 frame_rate: float = 1.0,
                 noise_intensity: float = 0.001,
                 appearance_momentum: float = 0.9,
                 ):
        """
        Initializes the DeepSORT tracker.

        Args:
            confirmation_threshold: A tuple (T_confirm_low, T_confirm_high) representing the
                                   number of consecutive frames a track must be assigned to be
                                   considered confirmed.
            deletion_threshold: A tuple (T_delete_low, T_delete_high) representing the number
                                of consecutive frames a track can be missed before being deleted.
            appearance_update: Method for updating appearance features: "Gallery", "EMA", "EMA Gallery".
            max_num_appearance_frames: Maximum number of appearance frames to store in the gallery
                                       (only used when appearance_update is "Gallery" or "EMA Gallery").
            mahalanobis_assignment_threshold: Threshold for the Mahalanobis distance.
            appearance_assignment_threshold: Threshold for the cosine distance between appearance features.
            iou_assignment_threshold: Threshold for the IoU distance.
            appearance_weight: Weight (lambda) for combining Mahalanobis and cosine distances.
            frame_size: The size of the video frame (width, height).
            frame_rate: The frame rate of the video.
            noise_intensity: The noise intensity for the Kalman filter.
            appearance_momentum: Momentum parameter for Exponential Moving Average (EMA) update.
        """

        self.confirmation_threshold = confirmation_threshold
        self.deletion_threshold = deletion_threshold
        self.appearance_update = appearance_update
        self.max_num_appearance_frames = max_num_appearance_frames
        self.mahalanobis_assignment_threshold = mahalanobis_assignment_threshold
        self.appearance_assignment_threshold = appearance_assignment_threshold
        self.iou_assignment_threshold = iou_assignment_threshold
        self.appearance_weight = appearance_weight
        self.frame_size = frame_size
        self.frame_rate = frame_rate
        self.noise_intensity = noise_intensity
        self.appearance_momentum = appearance_momentum

        self.tracks = []  # List to store track objects
        self.next_track_id = 1  # Assign unique IDs to tracks

        self.measurement_noise = np.diag([25, 25, 25, 25]) #Constant, the example
        self.kf = self._create_kalman_filter() #Just one Kalman Filter for simplicity.

    def _create_kalman_filter(self):
        """
        Creates a Kalman filter object.  You will need to adapt this to your
        specific Kalman filter implementation.  This is a placeholder.
        """
        class DummyKalmanFilter:
            def __init__(self, noise_intensity):
                self.noise_intensity = noise_intensity
            def predict(self, state, covariance):
                # Placeholder predict function
                state[4:] += self.noise_intensity # Update the speed terms
                return state, covariance + np.diag([self.noise_intensity]*8)
            def update(self, state, covariance, measurement):
                # Placeholder update function.  Calculate the innovation, etc. in a real implementation
                return state, covariance
        return DummyKalmanFilter(self.noise_intensity)

    def predict_and_update_tracks(self,frame_width, frame_height):
        """
        Predicts the next state of each track using the Kalman filter and removes
        tracks that are outside the frame.
        """
        for track in self.tracks:

            track['state'], track['state_covariance'] = self.kf.predict(track['state'], track['state_covariance'])

            # Clip the bounding box to stay within frame bounds. Be careful with this.
            track['state'][0] = np.clip(track['state'][0], 0, frame_width)
            track['state'][1] = np.clip(track['state'][1], 0, frame_height)


    def assign_detections_to_tracks(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Assigns detections to existing tracks using the matching cascade.

        Args:
            detections: A list of detection dictionaries.

        Returns:
            A tuple containing:
                - A list of (track_index, detection_index) tuples representing assigned pairs.
                - A list of track indices that were not assigned.
                - A list of detection indices that were not assigned.
        """

        # 1. Gating based on Mahalanobis distance and appearance cosine distance
        gated_cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                mahalanobis_dist = mahalanobis_distance(track, detection, self.measurement_noise)
                cosine_dist = cosine_distance(track, detection, self.appearance_momentum)

                # Gating
                if mahalanobis_dist > self.mahalanobis_assignment_threshold:
                    mahalanobis_dist = float('inf')
                if cosine_dist > self.appearance_assignment_threshold:
                    cosine_dist = float('inf')

                gated_cost_matrix[i, j] = self.appearance_weight * mahalanobis_dist + \
                                          (1 - self.appearance_weight) * cosine_dist

        # 2. Matching cascade
        assigned_tracks = []
        unassigned_tracks = list(range(len(self.tracks)))
        unassigned_detections = list(range(len(detections)))

        # Split tracks into groups based on time since last assignment
        max_age = max(track['time_since_update'] for track in self.tracks) if self.tracks else 0
        track_groups = [[] for _ in range(max_age + 1)]
        for track_index in unassigned_tracks:
            track = self.tracks[track_index]
            track_groups[track['time_since_update']].append(track_index)

        for age in range(max_age + 1):
            if not track_groups[age]:
                continue

            track_indices = track_groups[age]

            # Create cost matrix for this age group
            cost_matrix = gated_cost_matrix[track_indices][:, unassigned_detections]

            # Perform linear assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                track_index = track_indices[r]
                detection_index = unassigned_detections[c]
                assigned_tracks.append((track_index, detection_index))

                # Remove assigned tracks and detections from unassigned lists
                if track_index in unassigned_tracks:
                    unassigned_tracks.remove(track_index)
                if detection_index in unassigned_detections:
                    unassigned_detections.remove(detection_index)

        # 3. IoU assignment for remaining unassigned tracks and detections
        iou_cost_matrix = np.zeros((len(unassigned_tracks), len(unassigned_detections)), dtype=np.float32)
        for i, track_index in enumerate(unassigned_tracks):
            track = self.tracks[track_index]
            for j, detection_index in enumerate(unassigned_detections):
                detection = detections[detection_index]
                iou_cost_matrix[i, j] = 1 - iou(track['state'][:4], detection['bbox'])  # 1 - IoU

        row_ind, col_ind = linear_sum_assignment(iou_cost_matrix)

        for r, c in zip(row_ind, col_ind):
            track_index = unassigned_tracks[r]
            detection_index = unassigned_detections[c]
            if iou_cost_matrix[r, c] < (1 - self.iou_assignment_threshold):  # iou > threshold, then assign
                assigned_tracks.append((track_index, detection_index))
                unassigned_tracks.remove(track_index)
                unassigned_detections.remove(detection_index)

        return assigned_tracks, unassigned_tracks, unassigned_detections

    def update_tracks(self, assigned_tracks: List[Tuple[int, int]], detections: List[Dict]):
        """
        Updates track states with assigned detections, handles unassigned tracks,
        and updates appearance features.

        Args:
            assigned_tracks: A list of (track_index, detection_index) tuples.
            detections: A list of detection dictionaries.
        """

        for track_index, detection_index in assigned_tracks:
            track = self.tracks[track_index]
            detection = detections[detection_index]

            # Update track state using Kalman filter (placeholder)
            track['state'], track['state_covariance'] = self.kf.update(track['state'], track['state_covariance'], detection['bbox'])
            track['time_since_update'] = 0
            track['hits'] += 1
            track['misses'] = 0

            # Update appearance features
            if self.appearance_update == "Gallery":
                if 'appearance_history' not in track:
                    track['appearance_history'] = deque(maxlen=self.max_num_appearance_frames)
                track['appearance_history'].append(detection['appearance'])
            elif self.appearance_update == "EMA":

                if 'appearance_history' not in track:
                    track['appearance_history'] = detection['appearance'] #Initialize
                else:

                    track['appearance_history'] = (self.appearance_momentum * track['appearance_history'] +
                                                 (1 - self.appearance_momentum) * detection['appearance'])

                    track['appearance_history'] = track['appearance_history'] / torch.linalg.norm(track['appearance_history'])
            elif self.appearance_update == "EMA Gallery":
                if 'appearance_history' not in track:
                    track['appearance_history'] = deque(maxlen=self.max_num_appearance_frames)
                    track['appearance_history'].append(detection['appearance']) #Initialize
                else:

                    ema_appearance = (self.appearance_momentum * track['appearance_history'][-1] +
                                                 (1 - self.appearance_momentum) * detection['appearance'])

                    ema_appearance = ema_appearance / torch.linalg.norm(ema_appearance)
                    track['appearance_history'].append(ema_appearance)

    def handle_unassigned_tracks(self, unassigned_tracks: List[int]):
        """
        Handles unassigned tracks by incrementing their time_since_update counter
        and deleting tracks that have been unassigned for too long.

        Args:
            unassigned_tracks: A list of track indices that were not assigned.
        """
        for track_index in unassigned_tracks:
            track = self.tracks[track_index]
            track['time_since_update'] += 1
            track['misses'] += 1

    def delete_lost_tracks(self):
        """
        Deletes tracks that have been unassigned for longer than the deletion threshold.
        """
        self.tracks = [track for track in self.tracks if track['time_since_update'] <= self.deletion_threshold[1]]

    def create_new_tracks(self, unassigned_detections: List[int], detections: List[Dict]):
        """
        Creates new tracks for unassigned detections.

        Args:
            unassigned_detections: A list of detection indices that were not assigned.
            detections: A list of detection dictionaries.
        """
        for detection_index in unassigned_detections:
            detection = detections[detection_index]

            # Initialize track state (replace with your Kalman filter initialization)
            state = np.array(detection['bbox'] + [0, 0, 0, 0], dtype=np.float32)  # x, y, w, h, vx, vy, vw, vh
            covariance = np.eye(8) * 10  # Initial covariance

            new_track = {
                'id': self.next_track_id,
                'state': state,
                'state_covariance': covariance,
                'time_since_update': 0,
                'hits': 1,
                'misses': 0,
                'age': 1, #How long has this track been alive
                'bbox':detection['bbox'] #Store the bounding box for visualization
            }
            self.tracks.append(new_track)
            self.next_track_id += 1

    def run(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Runs the DeepSORT tracker on a single frame.

        Args:
            frame: The current video frame (NumPy array).
            detections: A list of detection dictionaries.

        Returns:
            A list of track dictionaries, containing the updated track information.
        """

        self.predict_and_update_tracks(frame.shape[1],frame.shape[0]) #Use the shape, not the attribute

        assigned_tracks, unassigned_tracks, unassigned_detections = self.assign_detections_to_tracks(detections)

        self.update_tracks(assigned_tracks, detections)
        self.handle_unassigned_tracks(unassigned_tracks)
        self.delete_lost_tracks()
        self.create_new_tracks(unassigned_detections, detections)

        # Tentative tracks deletion. Tracks will start to be confirmed after Confirmation_threshold.
        confirmed_tracks = [track for track in self.tracks if track['hits'] >= self.confirmation_threshold[0]]

        #Tentative tracks visualization, can be empty
        tentative_tracks = [track for track in self.tracks if track['hits'] < self.confirmation_threshold[0]]

        # Delete out-of-frame tracks (using the helper function)
        # This can be adapted to delete by age, #misses, etc.
        delete_out_of_frame_tracks(self, confirmed_tracks, frame.shape[1], frame.shape[0]) #Modified to return tracker


        return self.tracks, tentative_tracks, len(self.tracks)

    def delete_track(self, track_id: int):
        """
        Deletes a track with the given ID.

        Args:
            track_id: The ID of the track to delete.
        """
        self.tracks = [track for track in self.tracks if track['id'] != track_id]


# --- Main Script ---
import random

if __name__ == "__main__":
    """
    Main script to run the DeepSORT tracker.
    """
    # 0. Configuration

    # Replace with the actual paths to your video and detections file
    video_path = "PedestrianTrackingVideo.avi"  # Replace with your video file
    detections_path = "PedestrianTrackingYOLODetections.pth" #Assumed to be a pickle.

    # 1. Load ReID Network
    download_reid_resnet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available

    #Replace with your code to load the weights, for example:
    net = torch.load("personReIDResNet_v2.pth", map_location=device)
    net.eval()

    # 2. Load Detections and Video
    # Create dummy detections for testing
    def create_dummy_detections(frame_width, frame_height, num_detections=5):
        detections = []
        for _ in range(num_detections):
            x = random.randint(0, frame_width - 50)
            y = random.randint(0, frame_height - 50)
            w = random.randint(20, 50)
            h = random.randint(30, 60)
            score = random.uniform(0.6, 0.99)  # Random score between 0.6 and 0.99
            detections.append({'bbox': [x, y, w, h], 'score': score}) #Added score
        return detections

    #video_path = 0 #To use the webcam

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video stream or file")
            exit()

    except Exception as e:
        print(f"Error loading video: {e}")
        print("Trying to access the webcam.")
        cap = cv2.VideoCapture(0)  # Try the webcam

        if not cap.isOpened():
            print("Error opening webcam.  Exiting.")
            exit()


    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize detections list
    detections = []

    try:
        # Load the detections from the .pth file
        detections = torch.load(detections_path)  # Load the entire dictionary.
        # Check if detections is a dictionary and has the key 'detections', if it does, extract the value of the key.
        if isinstance(detections, dict) and 'detections' in detections:
             detections = detections['detections']
        else:
            print("Warning: Detections file not found. Using dummy detections.")
            # Use create_dummy_detections to create dummy detections
            detections = create_dummy_detections(frame_width, frame_height)

    except FileNotFoundError:
        print("Warning: Detections file not found. Using dummy detections.")
        # Use create_dummy_detections to create dummy detections
        detections = create_dummy_detections(frame_width, frame_height)

    except Exception as e:
        print(f"Error loading or creating dummy detections: {e}")
        detections = create_dummy_detections(frame_width, frame_height) #Try anyway

    # 3. Initialize Tracker

    # Initialize the DeepSORT tracker with desired parameters
    tracker = DeepSORT(
        confirmation_threshold=(2, 2),
        deletion_threshold=(5, 5),
        appearance_update="Gallery",
        max_num_appearance_frames=50,
        mahalanobis_assignment_threshold=10.0,
        appearance_assignment_threshold=0.4,
        iou_assignment_threshold=0.95,
        appearance_weight=0.02,
        frame_size=(frame_width, frame_height), # Use the right frame size
        frame_rate=30.0,  # Adjust to your video's frame rate
        noise_intensity=0.001,
        appearance_momentum=0.9
    )

    # 4. Tracking Loop
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break

        frame_number += 1

        # Filter detections by score (assuming detections have a 'score' key)
        filtered_detections = [d for d in detections if d['score'] > 0.5]

        # 5. Run ReID Network
        filtered_detections = run_reid_net(net, frame, filtered_detections)

        # 6. Run Tracker
        tracks, tentative_tracks, total_tracks = tracker.run(frame, filtered_detections)

        # 7. Visualize Results

        # Annotate the frame with bounding boxes and track IDs
        for track in tracks:
            bbox = track['state'][:4]  # Get bounding box from Kalman filter state
            x, y, w, h = map(int, bbox)
            track_id = track['id']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show Tentative tracks with a different color
        for track in tentative_tracks:
            bbox = track['state'][:4]  # Get bounding box from Kalman filter state
            x, y, w, h = map(int, bbox)
            track_id = track['id']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {track_id} (Tentative)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("DeepSORT Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 8. Clean up
    cap.release()
    cv2.destroyAllWindows()
