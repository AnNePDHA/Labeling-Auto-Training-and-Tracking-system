import os
import time

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv
from threading import Thread, Event
from azure.ai.ml import MLClient, command, Input
from azure.identity import ClientSecretCredential
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.servicebus import ServiceBusClient, ServiceBusMessage
# from flask_cors import CORS, cross_origin

from model_func import *
from blob_storage_func import *
from tracking_func import *
from tracker import Tracker

# Load environment variables from .env file
load_dotenv()

# Extract configuration account Azure Machine Learning
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP")
WORKSPACE_NAME = os.environ.get("AZURE_WORKSPACE_NAME")

# Extract configuration Env, Compute, Model, Datastore, Experiment AML
AML_MODEL_NAME = os.environ.get("AML_MODEL_NAME")
AML_COMPUTE = os.environ.get("AML_COMPUTE")
AML_ENV = os.environ.get("AML_ENV")
AML_DATASTORE = os.environ.get("AML_DATASTORE")
AML_EXPERIMENT_NAME = os.environ.get("AML_EXPERIMENT_NAME")

# Extract configuration App Registrations
TENANT_ID = os.environ.get("AZURE_TENANT_ID")
CLIENT_ID = os.environ.get("AZURE_CLIENT_ID")
CLIENT_SECRET = os.environ.get("AZURE_CLIENT_SECRET")

# Extract configuration authentication
VALID_API_KEY = os.environ.get("API_AUTH_KEY")
URL_API_RECEIVER = os.environ.get("URL_API_RECEIVER")
URL_API_TRACKING_RECEIVER = os.environ.get("URL_API_TRACKING_RECEIVER")

# Extract configuration storage paths
DATASTORE_ASSET = os.environ.get("DATASTORE_ASSET")
PROCESS_PATH = os.environ.get("PROCESS_TRACKING_PATH")
SAVE_TRAINING_PATH = os.environ.get("SAVE_TRAINING_PATH")

# Extract Azure Service Bus connection string, topic name, and subscription name
SVBUS_CONNECTION_STR = os.environ.get("SVBUS_CONNECTION_STR")
RECEIVE_SUBSCRIPTION_NAME = os.environ.get("RECEIVE_SUBSCRIPTION_NAME")
DEV_TOPIC_NAME = os.environ.get("DEV_TOPIC_NAME")
DA_QUEUE_NAME = os.environ.get("DA_QUEUE_NAME")

with open('training-config/train-config.yaml', 'r') as file:
        config = yaml.safe_load(file)

CLASS_NAMES = config.get('names', [])
NEW_RESULTS_TRACKING = {}
TRACKER_LST = {}
# PROCESS_PATH = "BF-training-data/process_tracking"
END_FLAG = True
RUN_FLAG = False

# Declare global run_id
RUN_ID = None

###################################################################################
############################# SYSTEM FUNCTIONS ####################################

def trigger_error(msg_error, error_type):
    """
    Triggers a custom error response with a provided message and error code

    Inputs:
        msg_error (str): The custom error message to be included in the response
        error_type (str): The type status to be returned with the error

    Outputs:
        Raise error based on error type and msg error
    """
    if error_type == 'ValueError':
        raise ValueError(msg_error)
    elif error_type == 'TypeError':
        raise TypeError(msg_error)
    elif error_type == 'KeyError':
        raise KeyError(msg_error)
    elif error_type == 'FileNotFoundError':
        raise FileNotFoundError(msg_error)
    elif error_type == 'RuntimeError':
        raise RuntimeError(msg_error)
    elif error_type == 'AttributeError':
        raise AttributeError(msg_error)
    else:
        raise Exception(msg_error)
 


def work_training(ml_client, model_name, latest_version, start_timedate, lst_blobs, img_lst_blobs):
    """
    Initialling process auto training in background
    """
    try:
        # Configure Job to know what datastore, model, environment, compute and experiment in Azure Machine Learning
        job = command(
            # Configure Datastore and Model
            inputs=dict(
                training_data=Input(
                    type="uri_folder",
                    path=f"azureml:{AML_DATASTORE}",
                ),
                model_to_train=Input(
                    type="custom_model",
                    path=f"azureml:{model_name}:{latest_version}"
                )
            ),

            # Where to save file .yaml config training data
            code="training-config",

            # Command to run in terminal to start train Yolo model
            command="""
                sed -i "s|path:.*$|path: ${{ inputs.training_data }}|" train-config.yaml &&
                yolo task=detect train data=train-config.yaml model=${{ inputs.model_to_train }} batch=1 epochs=2 lr0=0.001 lrf=0.001 patience=1 project=yolov8-cherry-detection name=yolov8_test_noop
            """,

            # Configure Environment
            environment= f"azureml:{AML_ENV}",

            # Configure Compute instance 
            compute= AML_COMPUTE,
            
            # Display name of Job run in experiment
            display_name= AML_EXPERIMENT_NAME,

            # Configure job belong to what experiment
            experiment_name= AML_EXPERIMENT_NAME
        )
        # Call global RUN_ID
        global RUN_ID

        # Start running job
        run_job = ml_client.create_or_update(job)
        run_id = run_job.component.name # Get run_id of the job

        # Assign RUN_ID global when success running job
        RUN_ID = run_id

        previous_status = "Starting"
        send_update_api(run_id, previous_status, URL_API_RECEIVER)

        # Check if that run is complete or not
        while True:
            # Get the run details
            latest_run = ml_client.jobs.get(run_id)
            
            # Check the status of the run
            run_status = latest_run.status
            # print(f"Run '{run_id}' status: {run_status}")

            # Check if status training change or not
            if run_status != previous_status:
                send_update_api(run_id, run_status, URL_API_RECEIVER)
                previous_status = run_status

            if run_status in ["Completed", "Failed", "Canceled"]:
                RUN_ID = None
                break
        ##################################################################################################
        ################################# Update Neu Model and Blob storage ##############################
        # Saving location
        save_foder = f'Training_{start_timedate}' # 

        # If training success update new model and move all training dataset out of folder
        if run_status == "Completed":
            # Update new model into Azure Machine Learning
            new_model = Model(
                path=f"azureml://jobs/{run_id}/outputs/artifacts/paths/weights/best.pt",
                name = model_name,
                description="cherry model recognition",
                type=AssetTypes.CUSTOM_MODEL,
            )

            ml_client.models.create_or_update(new_model)    # Upload this model to Azure Machine Learning

            ml_client.models.download(name=model_name, version=new_model.version) # Download this model to local
            print(f"Update {model_name} version {new_model.version} downloaded.")

            # Move all labels folder to save path
            for blob_path in lst_blobs:
                file_name = blob_path.split("/")[-1]
                move_blob(blob_path, f'{DATASTORE_ASSET}/{SAVE_TRAINING_PATH}/{save_foder}/labels/{file_name}')

            # Move all images folder to save path
            for img_path in img_lst_blobs:
                file_name = img_path.split("/")[-1]
                move_blob(img_path, f'{DATASTORE_ASSET}/{SAVE_TRAINING_PATH}/{save_foder}/images/{file_name}')

        print("FINISH TRAINING")
        ##################################################################################################

    except Exception as e:
        print(f"An error occurred in 'work_training()': {str(e)}")
        trigger_error(f"An error occurred in 'work_training()': {str(e)}", "RuntimeError")
        

def run_training():
    """
    Initiates the training process.
    """
    ######################## Check and convert all label into yolo format ############################
    try:
        # Get all blob path in images folder and labels folder
        lst_blobs = list_blobs_in_folder(folder_path= f'{DATASTORE_ASSET}/labels')
        img_lst_blobs = list_blobs_in_folder(folder_path= f'{DATASTORE_ASSET}/images')

        # Raise error when not having any image to train
        if img_lst_blobs == []:
            raise ValueError('No images found in the training folder')

        # Check every label txt file in labels folder
        for blob_path in lst_blobs:
            list_lines = read_file_from_blob(blob_path).decode('utf-8').splitlines()

            # Check if that label already convert to yolo format or not
            if len(list_lines[0].split(" ")) > 5:   # Check if there are more 5 data items in first line
                file_name = blob_path.split("/")[-1]
                new_content = []
                # print(file_name)
                w_img, h_img = None, None

                # Find image in images folder base on file name
                for img_path in img_lst_blobs:
                    if file_name[:-4] in img_path:  # Check if file name exist in images folder
                        img_data = read_file_from_blob(img_path)

                        # Convert the bytes data to a numpy array
                        image_array = np.frombuffer(img_data, np.uint8)

                        # Decode the array into an image
                        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                        # Get shape of image
                        w_img, h_img, _ = image.shape
                
                # If file image not exist raise error
                if w_img == None:
                    raise FileNotFoundError(f"The file {file_name} image was not found in the folder images")

                # Convert line to line into yolo format
                for line in list_lines:
                    lst_content = line.split(" ")
                    new_line = convert2yololabel(lst_content[0], lst_content[1:], w_img, h_img)
                    new_content.append(new_line)
                
                # Combine all line and rewrite into that file
                new_content = "\n".join(new_content)
                rewrite_blob_content(blob_path, new_content)
    except Exception as e:
        print(f"An error occurred in preprocessing: {str(e)}") 
        return trigger_error(f"An error occurred in preprocessing: {str(e)}", 'RuntimeError')
        
    ##################################################################################################
    ############################ Training Job on Azure Machine Learning ##############################
    # Let's login to configure your workspace and resource group.
    credential = ClientSecretCredential(
        tenant_id=TENANT_ID,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )

    # Get a handle to the workspace. You can find the info on the workspace tab on ml.azure.com
    ml_client = MLClient(
        credential=credential,
        subscription_id = SUBSCRIPTION_ID,
        resource_group_name = RESOURCE_GROUP,
        workspace_name = WORKSPACE_NAME,
    )

    
    # Name of model in Azure Machine Learning
    model_name = AML_MODEL_NAME
    # List all versions of the specified model
    models = ml_client.models.list(name= model_name)
    # Sort models by version and get the latest one
    latest_model = max(models, key=lambda m: int(m.version))
    latest_version = latest_model.version

    print(f'Get latest model "{model_name}" version {latest_version}')

    # Get the current time and date
    start_timedate = time.localtime()

    # Format the current time and date
    formatted_time_date = time.strftime("%Y-%m-%d %H:%M:%S", start_timedate)

    # Start the background task in a new thread with the arguments
    thread = Thread(target=work_training, args=(ml_client, model_name, latest_version, formatted_time_date, lst_blobs, img_lst_blobs), daemon= True)
    thread.start()

    results = {
        'message' : f"RUN TRAINING SUCCESS! MODEL_NAME '{model_name}' - VERSION {latest_version}",
        'startTime' : formatted_time_date
    }
    

    return results

def trainingprogress():
    """
    Retrieves the progress of the ongoing training.
    """
    
    # Let's login to configure your workspace and resource group.
    credential = ClientSecretCredential(
        tenant_id=TENANT_ID,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )

    # Get a handle to the workspace. You can find the info on the workspace tab on ml.azure.com
    ml_client = MLClient(
        credential=credential,
        subscription_id = SUBSCRIPTION_ID,
        resource_group_name = RESOURCE_GROUP,
        workspace_name = WORKSPACE_NAME,
    )

    if RUN_ID != None:
        # Get the run details
        latest_run = ml_client.jobs.get(RUN_ID)
        
        # Check the status of the run
        run_status = latest_run.status

        results = {
            "messages" : "Job is running",
            "run_id" : RUN_ID,
            "progress" : run_status
        }
    else:
        results = {
            "messages" : "No job run",
            "run_id" : None,
            "progress" : None
        }
    
    return results


def post_model(input_data):
    """
    Receives and processes a model file sent to the server.
    """
    # Get input url file
    results = None
    detect_lst = []

    # Check if 'img_url' exits or not
    if "img_url" not in input_data:
        results = {
            'img_url' : None,
            'detection' : detect_lst
        }
    else: 
        # Base on img_url, recognition cherry in this image
        img_url = input_data["img_url"] 

        if 'https://' in img_url:
            img = readImg_byurl(img_url)
        else:
            img = cv2.imread(img_url)
        # Get height, width of image
        height, width, _ = img.shape

        # print(img_url)
        res_lst = cherryclassification(img)

        # Append result for each cherry
        for res in res_lst:
            x, y, w, h = res['x'], res['y'], res['w'], res['h']

            #Convert yolo label to web foramt
            bbox = yolo2webbox(x, y, w, h, width, height)
            detect_lst.append({
                'bbox' : bbox,
                'conf_score' : res['conf'],
                'class_name' : CLASS_NAMES[res['label']]
            })
        
        results = {
            'img_url' : img_url,
            'detection' : detect_lst
        }


    return results

def work_tracking(progress_type, batch_id, tracker, video_url, update_time_sc, video_ord):
    """
    Initialling process tracking video in background
    """
    global NEW_RESULTS_TRACKING
    global TRACKER_LST
    global END_FLAG
    global RUN_FLAG
    # Define run flag
    RUN_FLAG = True

    try:
        if video_url:
            # Extract the file name from the path
            file_name = os.path.basename(video_url)

            # Get file blob path from video_url
            file_blob_path = video_url.split(f'{CONTAINER_NAME}/')[-1]

            # Move file to tracking_process folder
            move_blob(file_blob_path, f"{DATASTORE_ASSET}/{PROCESS_PATH}/{batch_id}/{file_name}")

            # Waiting to process
            l_video = len(NEW_RESULTS_TRACKING[batch_id]["videos"])
            if video_ord > l_video:
                print("WAITING...")
            while video_ord > l_video:
                time.sleep(5)
                l_video = len(NEW_RESULTS_TRACKING[batch_id]["videos"])

            # Download video to local file
            local_file_path = f"tracking_process/{batch_id}.mp4"
            with open(local_file_path, "wb") as download_file:
                download_file.write(read_file_from_blob(f"{DATASTORE_ASSET}/{PROCESS_PATH}/{batch_id}/{file_name}"))

            print("PROCESSS TRACKING VIDEO")
            # Process video to get every ID cherry and time of video
            res_tracking = process_video(local_file_path, model, tracker)

            # Process all new cherry ID of video
            update_dict = {}
            for id, info in res_tracking.items():
                # print(f"{id} : {info['last_area']}")
                area = info['last_area']
                update_dict.update({str(id): [info['last_area'], str(convert_size_row(area))]})

            # Extract video_id
            new_video_id = "id_" + str(video_ord)

            # Update new result tracking
            RESULTS_VIDEO = {
                "video_id" : new_video_id,
                "record_time" : update_time_sc,
                "id_stats" : update_dict,
            }
            NEW_RESULTS_TRACKING[batch_id]["videos"].append(RESULTS_VIDEO)
            print(f"FINISHED TRACKING VIDEO {video_ord}")
            # results = jsonify([RESULTS_TRACKING])

        if progress_type == 'end':
            # print("END TRACKING")

            # Waiting to process
            l_video = len(NEW_RESULTS_TRACKING[batch_id]["videos"])
            while video_ord > l_video:
                print("WAITING...")
                time.sleep(5)
                l_video = len(NEW_RESULTS_TRACKING[batch_id]["videos"])

            # batch_id = list(NEW_RESULTS_TRACKING.keys())[-1]
            video_stats_lst = NEW_RESULTS_TRACKING[batch_id]["videos"]
            total_time_record = 0
            cherry_stats = {}

            for video_stats in video_stats_lst:
                time_record = video_stats["record_time"]
                id_stats = video_stats["id_stats"]

                total_time_record += time_record
                cherry_stats.update(id_stats)
                
            # Update classify counts
            classify_count = classify_sizes(cherry_stats)
            RESULTS_TRACKING = {
                "batch_id" : batch_id,
                "record_time" : total_time_record,
                "start_date" : NEW_RESULTS_TRACKING[batch_id]["start_date"],
                "id_stats" : cherry_stats,
                "classify_count" : classify_count,
                "measure" : "Row",
                "total" : len(cherry_stats)
            }
            
            # Export the dictionary as JSON into Blob storage
            upload_json_to_blob(f"{DATASTORE_ASSET}/{PROCESS_PATH}/{batch_id}/{batch_id}.json", RESULTS_TRACKING)

            # Remove local video file path
            local_file_path = f"tracking_process/{batch_id}.mp4"
            os.remove(local_file_path)
            delete_webm_files_in_folder(f"{DATASTORE_ASSET}/{PROCESS_PATH}/{batch_id}/")
            # print(f"{local_file_path} has been deleted.")
            NEW_RESULTS_TRACKING.pop(batch_id)
            TRACKER_LST.pop(batch_id)
            RUN_FLAG = False

            # Response to the receiver the status of training
            data = RESULTS_TRACKING
            requests.post(URL_API_TRACKING_RECEIVER, json=data)
            # print(f"BACKGROUND PROCESS VIDEO ORD {video_ord} COMPLETED:")
            print(RESULTS_TRACKING)
      

    except Exception as e:
        # Clean up resources
        local_file_path = f"tracking_process/{batch_id}.mp4"
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        NEW_RESULTS_TRACKING.pop(batch_id, None)
        TRACKER_LST.pop(batch_id, None)
        delete_webm_files_in_folder(f"{DATASTORE_ASSET}/{PROCESS_PATH}/{batch_id}/")
        RUN_FLAG = False

        # Can start video again if process failed
        if progress_type == 'start':
            END_FLAG = True

        # Response to the receiver the status of training
        data = {
            "message" : f"An error occurred in 'work_tracking()': {str(e)}"
        }
        requests.post(URL_API_TRACKING_RECEIVER, json=data)
        print(f"An error occurred in 'work_tracking()': {str(e)}")

def tracking(input_data, key_run = None):
    """
    Executes the tracking process using the provided data
    """
    global NEW_RESULTS_TRACKING
    global TRACKER_LST
    global END_FLAG

    # Initial value
    video_url = None
    update_time_sc = 0
    
    # Check if 'video_url' exits or not
    if "video_url" not in input_data:
        if key_run != 'end':
            trigger_error("ERROR NOT HAVING VIDEO!", "FileNotFoundError")
           
        else:
            if END_FLAG:
                # print("ERROR NOT START VIDEO YET!")
                trigger_error("ERROR NOT START VIDEO YET!", "RuntimeError")
                # abort(404)
            # else:
            #     END_FLAG = True
    else:
        # Base on video_url, recognition cherry in this video
        video_url = input_data["video_url"] 

    # Check if 'duration' exits or not
    if ("duration" not in input_data) and ("video_url" in input_data):
        trigger_error("ERROR NOT HAVING VIDEO DURATION!", "AttributeError")
    elif "duration" in input_data: 
        # Get video duration 
        update_time_sc = input_data["duration"]


    # If key run is start, refresh init tracker and result tracking
    if key_run == 'start':
        # print('START TRACKING')
        if not END_FLAG:
            # print("ERROR NOT END VIDEO YET!")
            trigger_error("ERROR NOT END VIDEO YET!", "RuntimeError")
            # abort(404)
        END_FLAG = False

        # Initial new tracker
        new_tracker = Tracker()
        # new_batch_id = "batch_id_" + str(len(NEW_RESULTS_TRACKING)) 
    
        # Get the current time and date
        start_timedate = time.localtime()

        # Format the current time and date
        saved_time_date = time.strftime("%Y%m%d%H%M%S", start_timedate)
        formatted_time_date = time.strftime("%Y-%m-%d %H:%M:%S", start_timedate)

        # Generate new batch_id
        batch_id = "Tracking_" + saved_time_date

        # Create new result tracking
        NEW_RESULTS_TRACKING[batch_id] = {
            "start_date" : formatted_time_date,
            "videos" : []
        }
        # print("Update New Batch_id")
        TRACKER_LST[batch_id] = new_tracker

    else:
        # Check if END_FLAG already True so raise error 
        if END_FLAG:
            trigger_error("ERROR NOT START VIDEO YET!", "RuntimeError")
            # abort(404)

        # If not, change END_FLAG
        if key_run == 'end':
            END_FLAG = True
    
    # Get batch_id of video process
    batch_id = list(NEW_RESULTS_TRACKING.keys())[-1]
    # Get the order of processing
    video_ord = len(list_blobs_in_folder(folder_path= f"{DATASTORE_ASSET}/{PROCESS_PATH}/{batch_id}"))
    l_video = len(NEW_RESULTS_TRACKING[batch_id]["videos"])

    # Get tracker base on batch_id
    tracker = TRACKER_LST[batch_id]

    # Start the background task in a new thread with the arguments
    thread = Thread(target=work_tracking, args=(key_run, batch_id, tracker, video_url, update_time_sc, video_ord), daemon= True)
    thread.start()

    return {"message": f"TRACKING SUCCESS! VIDEO_ORD: {video_ord} - NO. PROCESS: {l_video}", "batch_id": batch_id}

def videoprogress():
    batch_id = None
    if RUN_FLAG:
        batch_id = list(NEW_RESULTS_TRACKING.keys())[0]
    
    return {
        "run_flag" : RUN_FLAG,
        "batch_id" : batch_id
    }

###################################################################################
######################## SERVICE BUS FUNCTIONS ####################################

# Event to signal the listener thread to stop
stop_event = Event()

def trigger_func_svbus(message, func_name, app_properties):
    """
    Processes the received message from Azure Service Bus

    Inputs:
    - message (dict): The content of the message in JSON format

    Outputs:
    - func(): return of the result function
    - func_name (str): name of subject/function trigger
    """
    # print("FUNC_NAME:", func_name)
    # Try to find the function in the global scope
    func = globals().get(func_name)

    if func is None:
        raise ValueError(f"Function '{func_name}' not found.")
    
    if message:
        # Convert message body to a string
        message_body = str(message)
        # print(f"Received message: {message_body}")

        # Parse the message body as a JSON dictionary
        json_data = json.loads(message_body)

    else: json_data = None
    if json_data or json_data == {}:
        if func_name == "tracking":
            key_run = None
            # print("FUNC - 2")
            # Extract user-defined properties (custom key-value pairs)
            key_run = app_properties.get("key_run", None)

            return func(json_data, key_run)
        
        # print("FUNC - 1")
        return func(json_data)
    
    # print("FUNC - 0")
    return func()


def send_message_to_topic(mess, func_name, mess_properties):
    """
    Sends a message to an Azure Service Bus topic.

    Inputs:
    - mess (str/dict): A plain text or dictionary containing the message content and any properties to be included in the message
    - label (str): name subject/label of the message
    """
    # Check if message is the dictionary
    if type(mess) is dict:
        content_message = json.dumps(mess)
    else: content_message = mess

    # Create a Service Bus client
    with ServiceBusClient.from_connection_string(SVBUS_CONNECTION_STR) as client:
        # Get the topic sender
        sender = client.get_topic_sender(topic_name= DEV_TOPIC_NAME)
        with sender:
            # Create a Service Bus message
            message = ServiceBusMessage(
                body = content_message,
                subject = func_name,
                application_properties = mess_properties
            )
            # Send the message to the topic
            sender.send_messages(message)
            print(f"Sending result to Dev server: '{content_message}' func name '{func_name}' with Custom properties: '{mess_properties}'")


def receive_messages_from_queue():
    """
    Receives messages from the specified Azure Service Bus queue.
    """
    # Create a Service Bus client using the connection string
    servicebus_client = ServiceBusClient.from_connection_string(conn_str=SVBUS_CONNECTION_STR)

    # Get a receiver to fetch messages from the queue
    with servicebus_client:
        receiver = servicebus_client.get_queue_receiver(queue_name=DA_QUEUE_NAME, max_wait_time=5)

        # Receive messages from the queue
        with receiver:
            print(f"Listening for messages on queue: {DA_QUEUE_NAME}...")
            while not stop_event.is_set(): 
                # Receive messages
                received_messages = receiver.receive_messages(max_message_count=10, max_wait_time=5)
                for msg in received_messages:
                    # Get name of function from subjact/label name message
                    func_name = msg.subject
                    # Extract user-defined properties (custom key-value pairs)
                    app_properties = msg.application_properties
                    # print("Properties:", app_properties)
                    if app_properties:
                        decoded_properties = {key.decode('utf-8'): value.decode('utf-8') for key, value in app_properties.items()}
                    else: decoded_properties = {}
                    # Try Except to catch error when trigger function
                    try:
                        # print(f"Received message: {str(msg)}")
                        # Process the message (here we just print it, but you can do any processing)
                        result_msg = trigger_func_svbus(msg, func_name, decoded_properties)
                        # print(result_msg)
                        # Complete the message so it's removed from the queue
                        receiver.complete_message(msg)
                        # print(f"Message processed and removed from queue.")
                        # Sending result messages to DEV receiver service bus
                        send_message_to_topic(result_msg, func_name, decoded_properties)

                    except Exception as e:
                        # Sending error messgage to Dev receiver service bus
                        error_msg = {
                            "error_type": type(e).__name__,  # Get the type of the error
                            "error_message": str(e)          # Get the error message
                        }
                        print(f"Error occurred: {e}")
                        send_message_to_topic(error_msg, func_name, decoded_properties)

                        # Dead-letter the message if there's an error
                        receiver.dead_letter_message(msg, reason="ProcessingFailed", error_description="Simulated processing error")
                        # print("Message dead-lettered.") # Leave message on the queue for later retry
