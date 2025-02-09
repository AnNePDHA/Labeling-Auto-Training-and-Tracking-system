from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np
 
class Tracker:
    def __init__(self):
        # Set the maximum cosine distance for appearance-based matching
        # Lower values make the tracker more strict in matching objects
        max_cosine_distance = 0.1
        nn_budget = None  # Maximum size of the appearance descriptor gallery
 
        # Path to the pre-trained model for feature extraction
        encoder_model_filename = './mars-small128.pb'
 
        # Create a nearest neighbor distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
       
        # Initialize the Deep SORT tracker
        self.tracker = DeepSortTracker(metric)
       
        # Initialize the feature encoder
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)
       
        # Dictionary to store the class ID for each track ID
        self.track_class_ids = {}
       
        # List to store the currently active tracks
        self.tracks = []
       
        # ID for the next new track
        self.next_id = 1  
       
        # Dictionary to count occurrences of each class ID
        self.class_counts = {}  
 
    def update(self, frame, detections):
        if len(detections) == 0:
            # If there are no detections, predict the next state of the tracks
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks()
            return
 
        # Convert detections to the format required by the Deep SORT tracker
        bboxes = np.asarray([d[:-2] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]  # Convert to (x, y, w, h) format
        scores = [d[-2] for d in detections]
        class_ids = [d[-1] for d in detections]
 
        # Generate appearance features for the detections
        features = self.encoder(frame, bboxes)
 
        # Create Detection objects for the Deep SORT tracker
        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))
 
        # Update the tracker with the current detections
        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks(class_ids)
 
    def update_tracks(self, class_ids=None):
        tracks = []
        for track in self.tracker.tracks:
            # Skip tracks that are not confirmed or haven't been updated recently
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()  # Get the bounding box in (x1, y1, x2, y2) format
            id = track.track_id
 
            # Assign class ID to the track if it doesn't have one
            if id not in self.track_class_ids:
                self.track_class_ids[id] = class_ids[0]  
                self.class_counts[class_ids[0]] = self.class_counts.get(class_ids[0], 0) + 1
 
            class_id = self.track_class_ids[id]
            tracks.append(Track(id, bbox, class_id))
 
        self.tracks = tracks
 
class Track:
    def __init__(self, id, bbox, class_id):
        # Initialize the track with an ID, bounding box, and class ID
        self.track_id = id
        self.bbox = bbox
        self.class_id = class_id