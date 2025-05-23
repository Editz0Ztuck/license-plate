import cv2
import numpy as np
import pytesseract
import time
import re
from collections import Counter
import requests
import socket
from datetime import datetime
import pytz

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_hostname():
    return socket.gethostname()

def get_iso_timestamp():
    # Get current time in Oslo timezone
    oslo_tz = pytz.timezone('Europe/Oslo')
    current_time = datetime.now(oslo_tz)
    
    # Format matching Go's reference time exactly: 2006-01-02T15:04:05.000-07:00
    # Ensure milliseconds are exactly 3 digits with leading zeros
    milliseconds = current_time.strftime('%f')[:3].zfill(3)
    
    # Format timezone offset to match Go's format (-07:00)
    offset = current_time.strftime('%z')
    formatted_offset = f"{offset[:3]}:{offset[3:]}"
    
    return f"{current_time.strftime('%Y-%m-%dT%H:%M:%S')}.{milliseconds}{formatted_offset}"

def send_to_server(plate_number):
    url = "http://10.1.120.143:8080/licenseplate"
    timestamp = get_iso_timestamp()
    data = {
        'plate': plate_number,
        'hostname': get_hostname(),
        'timestamp': timestamp
    }
    
    # Debug print the exact data being sent
    print("\nSending data to server:")
    print(f"URL: {url}")
    print(f"Data: {data}")
    print(f"Timestamp format: {timestamp}")
    
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            print(f"Successfully sent plate {plate_number} to server")
        else:
            print(f"Failed to send plate to server. Status code: {response.status_code}")
            print(f"Server response: {response.text}")  # Print server's error message
    except Exception as e:
        print(f"Error sending to server: {str(e)}")

def is_valid_license_plate(text):
    text = text.upper().strip()
    text = ''.join(e for e in text if e.isalnum())
    if len(text) < 5:
        return False
    has_letter = any(c.isalpha() for c in text)
    has_number = any(c.isdigit() for c in text)
    return has_letter and has_number

def detect_license_plate(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Find all contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        # Check if the contour has the right aspect ratio for a license plate
        if 2.0 <= aspect_ratio <= 5.0:
            # Get the region of interest
            roi = gray[y:y+h, x:x+w]
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Try to read the text
            try:
                text = pytesseract.image_to_string(roi, config='--psm 7')
                text = ''.join(e for e in text if e.isalnum())
                print(f"Detected text: {text}")  # Debug print
                
                if text and is_valid_license_plate(text):
                    return frame, text
            except Exception as e:
                print(f"Error reading text: {str(e)}")  # Debug print
                continue
    
    return frame, None

def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Camera started. Press 'q' to quit")
    
    # Initialize detection tracking
    detections = []
    last_detection = None
    detection_cooldown = 0.01  # very short cooldown
    last_detection_time = 0
    max_detections = 3  # only need 3 detections
    last_sent_time = 0  # Track last time a request was sent
    min_send_interval = 1.5  # Minimum seconds between sends

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Process the frame
        processed_frame, plate_text = detect_license_plate(frame)

        # Display the frame
        cv2.imshow('License Plate Detection', processed_frame)

        # If we detected a plate and enough time has passed since last detection
        current_time = time.time()
        if plate_text and (current_time - last_detection_time) > detection_cooldown:
            if plate_text != last_detection:  # Only add if different from last detection
                detections.append(plate_text)
                last_detection = plate_text
                last_detection_time = current_time
                print(f"Detection {len(detections)}/3: {plate_text}")  # Debug print
                
                # If we have enough detections and enough time has passed since last send
                if len(detections) >= max_detections and (current_time - last_sent_time) > min_send_interval:
                    # Count occurrences of each plate
                    plate_counts = Counter(detections)
                    most_common = plate_counts.most_common(1)[0]
                    print(f"Most common plate: {most_common[0]}")
                    
                    # Send to server
                    send_to_server(most_common[0])
                    last_sent_time = current_time
                    
                    # Clear the detections list and start over
                    detections = []
                    last_detection = None
                    print("Starting new collection...")  # Debug print

        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 