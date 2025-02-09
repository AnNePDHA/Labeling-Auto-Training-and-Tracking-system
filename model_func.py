import secrets
import string

import cv2
import ultralytics
import requests
import numpy as np
from ultralytics import YOLO

# Load custom YOLOv8 model
model = YOLO('./cherry-YOLOv8/best.pt')

def generate_token(length=32):
    """
    Generate a secure token.
    Args:
    - length (str): Length of the token to be generated.

    Returns:
    - token (str): A secure token as a string.
    """
    # Define the characters to use in the token
    alphabet = string.ascii_letters + string.digits
    
    # Generate the token
    token = ''.join(secrets.choice(alphabet) for _ in range(length))
    
    return token


def yolo2webbox(x, y, w, h, img_width, img_height):
    """
    Convert YOLO bbox format (x, y, w, h) to 4 corner of the bbox.
    Args:
    - x, y: center coordinates of the bounding box (normalized)
    - w, h: width and height of the bounding box (normalized)
    - img_width, img_height: original image dimensions
    
    Returns:
    - bbox: list of 4 coordinates [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]
    """
    x_min = int((x - w / 2) * img_width)
    x_max = int((x + w / 2) * img_width)
    y_min = int((y - h / 2) * img_height)
    y_max = int((y + h / 2) * img_height)

    return [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]

def convert2yololabel(class_id, bbox, img_width, img_height):
    """
    Convert bbox format from coordinates (x1, y1, x2, y2, x3, y3, x4, y4) to YOLO format (class, x, y, w, h).
    Args:
    - class_id: class label
    - bbox: list of 4 coordinates [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]
    - img_width, img_height: original image dimensions
    
    Returns:
    - yolo_label: YOLO format label as string "class x y w h"
    """
    x_min, y_min = float(bbox[0]), float(bbox[1])
    x_max, y_max = float(bbox[4]), float(bbox[5])

    x = (x_min + x_max) / 2.0 / img_width
    y = (y_min + y_max) / 2.0 / img_height
    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_height

    return f"{class_id} {x} {y} {w} {h}"

def readImg_byurl(img_url):
    # Send a GET request to the image URL
    response = requests.get(img_url)

    # Raise an exception if the request was unsuccessful
    response.raise_for_status()

    # Convert the image to a NumPy array
    image_data = np.asarray(bytearray(response.content), dtype=np.uint8)

    # Decodes it into an image
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    return img

def cherryclassification(img):
    """
    Use a model to classify the image and return labels and bbox in the format (label, conf, x, y, w, h).
    Args:
    - model: trained model for cherry classification
    - img:  input image array reading by cv2
    
    Returns:
    - results: list of dictionaries with keys "label", "conf", "x", "y", "w", "h"
    """
    # Get height, width of image
    img_height, img_width, _ = img.shape

    # Run model from local and get results
    results = model(img)
    classifications = []

    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = r
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            conf = round(conf, 2)
            class_id = int(class_id)

            # Convert bbox to YOLO format
            x = (x1 + x2) / 2.0 / img_width
            y = (y1 + y2) / 2.0 / img_height
            w = (x2 - x1) / img_width
            h = (y2 - y1) / img_height

            classification = {
                "label": class_id,
                "conf": conf,
                "x": x,
                "y": y,
                "w": w,
                "h": h
            }
            classifications.append(classification)

    return classifications

def send_update_api(run_id, key_status, url_receiver):
    """
    Send an update on the progress of a job to a specified URL

    Inputs:
        key_status (str): The current status of the job
        url_receiver (str): The URL to which the update should be sent
    """
    # Percent progress of status
    map_progress ={
        "Starting" : 0,
        "Preparing" : 10,
        "Queued" : 20,
        "Running" : 40,
        "Finalizing" : 90,
        "Completed" : 100,
        "Failed" : 100,
        "Canceled" : 100
    }

    # Auto generate token progress
    # key_progress = generate_token()

    # Json data retunr
    data = {"run_id": run_id, "key_status": key_status, "progress": map_progress[key_status]}
    print(f"{key_status} : {map_progress[key_status]}%")

    # Response to the receiver the status of training
    requests.post(url_receiver, json=data)
    # if response.status_code == 200:
    #     print(f"Successfully sent update: {data}")
    # else:
    #     print(f"Failed to send update: {data}, Response Code: {response.status_code}")