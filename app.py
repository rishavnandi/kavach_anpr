# Import required libraries
from pathlib import Path
import streamlit as st
import pandas as pd
import requests
import tempfile
import time
import json
import cv2
import os

# Logo for the app
LOGO = 'https://img.icons8.com/color/256/null/opencv.png'

# Set parameters for Plate Recognizer API
PLATE_API = st.text_input(
    'Enter your Plate Recognizer API key', type='password')
REGIONS = ['in']

# Set parameters for Eden AI API
EDEN_API = st.text_input('Enter your Eden AI API key', type='password')

# Set output CSV file name
OUTPUT_CSV_FILE = 'detected_faces_and_plates.csv'

# Upload video file
VIDEO_FILE = st.file_uploader(
    'Upload a video file', type=['mp4'])

# Initialize set to store unique number plates and faces
unique_faces = set()
unique_plates = set()

with st.sidebar:
    st.image(LOGO)
    st.title('ANPR')
    st.subheader('Automatic Number Plate Recognition')
    st.write('Upload a video file to detect number plates and faces')


if VIDEO_FILE is not None:
    st.write('Video file uploaded successfully')
    st.video(VIDEO_FILE)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(VIDEO_FILE.getbuffer())
        VIDEO_FILE = temp_file.name

if st.button('Analyze'):
    with st.spinner('Analyzing video file...'):

        # Send video file to Eden AI API for face detection
        headers = {"Authorization": "Bearer " + EDEN_API}

        url = "https://api.edenai.run/v2/video/person_tracking_async"
        data = {"providers": "google"}
        files = {'file': open(VIDEO_FILE, 'rb')}

        response = requests.post(url, data=data, files=files, headers=headers)

        with open('faces.json', 'w') as f:
            json.dump(response.json(), f)

        # Parse response and job ID
        with open('faces.json') as f:
            response = json.load(f)
            job_id = response['public_id']

        unique_faces.add(job_id)

        st.info('Face detection completed')

        # Save all frames to a folder
        frames_folder = 'frames'
        if not os.path.exists(frames_folder):
            os.makedirs(frames_folder)

        # Read video file
        video_capture = cv2.VideoCapture(VIDEO_FILE)

        # Save all frames to a folder
        count = 0
        while True:
            ret, frame = video_capture.read()
            if ret:
                cv2.imwrite('frames/frame%d.jpg' % count, frame)
                count += 1
            else:
                break

        # Send request to Plate Recognizer API for each frame to detect number plates
        for i in range(count):
            response = requests.post(
                'https://api.platerecognizer.com/v1/plate-reader/',
                data=dict(regions=REGIONS),
                files=dict(upload=('filename', open(
                    'frames/frame%d.jpg' % i, 'rb'))),
                headers={'Authorization': f'Token {PLATE_API}'})

            # Save response to JSON file
            with open('plates.json', 'w') as f:
                json.dump(response.json(), f)

            # Parse response and extract number plates
            with open('plates.json') as f:
                response = json.load(f)
                for result in response['results']:
                    plate = result['plate']
                    unique_plates.add(plate)

        st.info('Number plate detection completed')

        # Save faces and number plates to CSV file

        # make sure both lists have the same length
        max_length = max(len(unique_faces), len(unique_plates))
        unique_faces = list(unique_faces)
        unique_plates = list(unique_plates)
        unique_faces += [None] * (max_length - len(unique_faces))
        unique_plates += [None] * (max_length - len(unique_plates))

        # create DataFrame and fill missing values
        df = pd.DataFrame({'faces_job_id': list(unique_faces), 'plates': list(
            unique_plates)}).fillna(method='ffill')

        st.info('CSV file created')
        st.success('Analysis completed')

        # Let user download CSV file

        st.download_button(
            label='Download CSV file',
            data=df.to_csv(),
            file_name=OUTPUT_CSV_FILE,
            mime='text/csv'
        )
