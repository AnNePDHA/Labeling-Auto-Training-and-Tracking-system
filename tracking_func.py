import cv2
import numpy as np

from ultralytics import YOLO
from flask import Flask, request, jsonify

# Load the YOLO model
model = YOLO("./deep_sort/models/cherry_best_ver2_4_1.pt")


def classify_sizes(input_dict):
    """
    Classify the sizes based on the input dictionary of areas.

    Inputs:
    - input_dict (dict): A dictionary where keys are identifiers (strings) and values are lists containing numeric areas.      
    Return:
    - classify_counts (dict): A dictionary with the counts of each unique area value.
    """
    # Declare result dict
    classify_counts = {}
    
    # Count appear times of each values in input_dict
    for values in input_dict.values():
        # Get the size value of each ID
        size_value = values[1]

        # Classify and append into classify_counts dict
        if size_value in classify_counts:
            classify_counts[size_value] += 1
        else:
            classify_counts[size_value] = 1
                
    return classify_counts

def merge_and_sum_dicts(target_dict, source_dict):
    """
    Append contents of source_dict to target_dict. If a key already exists in target_dict,
    its value will be summed with the value from source_dict.
    
    Inputs:
    - target_dict (dict): The dictionary to which contents will be appended.
    - source_dict (dict): The dictionary from which contents will be taken.
    Return:
    - target_dict (dict): The updated target_dict with merged and summed contents.
    """
    # Get key and value to combine from source_dict
    for key, value in source_dict.items():
        if key in target_dict:
            target_dict[key] = target_dict[key] + value
        else:
            target_dict[key] = value
    return target_dict

def convert_size_row(area):
    """
    Convert the given area into a different size row format.

    Inputs:
    - area (float): A numeric value representing an area
                 
    Return:
    - float : A numeric value representing after calculating.
    """
    size_row = (- area * 1.0 / 2535) + 17
    return round(size_row * 2) / 2

def process_video(video_path, model, tracker):
    """
    Process a video file, applying a model for detection and a tracker for tracking objects.

    Inputs:
    - video_path (str): A string representing the path to the video file.
                       Example: "path/to/video.mp4"
    - model (YOLO): An object representing the detection model to be applied to each frame.
                  Example: SomeModelObject
    - tracker (Tracker): An object representing the tracker to track detected objects across frames.
                    Example: SomeTrackerObject
                    
    Return:
    - track_class_area_dict (dict):A dictionary containing all detections across the video with frame numbers.
    - duration_seconds (int): A numeric value representing time of video
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
 
    # Get the total number of frames
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the frames per second (fps)
    # fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the duration in seconds
    # duration_seconds = int(total_frames / fps)

    track_class_area_dict = {}
 
    # Detection threshold
    detection_threshold = 0.6
 
    # Process the video frame by frame
    while ret:
        results = model(frame, verbose=False)
 
        detections = []
        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                if score > detection_threshold:
                    detections.append([x1, y1, x2, y2, score, class_id])
 
        # Update the tracker with the current frame and detections
        tracker.update(frame, detections)
 
        # Process each track in the tracker
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            class_id = track.class_id
 
            # Calculate the area of the bounding box
            width = x2 - x1
            height = y2 - y1
            area = round(width * height, 2)
            # area = [width, height]
            # Ensure the bounding box is within the frame boundaries
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue
 
            # Initialize or update the track information in the dictionary
            if track_id not in track_class_area_dict:
                track_class_area_dict[track_id] = {'class_id': class_id, 'areas': [], 'last_area': None}
 
            # Update the dictionary with the current area
            track_class_area_dict[track_id]['areas'].append(area)
            track_class_area_dict[track_id]['last_area'] = area

        ret, frame = cap.read()
 
 
    cap.release()

    return track_class_area_dict