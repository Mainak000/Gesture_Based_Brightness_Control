import cv2
import numpy as np
import mediapipe as mp
import time
import screen_brightness_control as sbc
from math import hypot
from comtypes import cast, POINTER


# Load the pre-trained Haar cascade for smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

def main():
   

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1, 
                          min_detection_confidence=0.75, min_tracking_confidence=0.75)
    draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)

    prev_left_distance = None
    lock_time = None
    brightness_locked = False
    locked_brightness = 0  # Store locked brightness level

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)
            left_landmark_list, right_landmark_list = get_left_right_landmarks(frame, processed, draw, mpHands)

            # Stop program if a smile is detected
            if detect_smile(frame):
                print("Smile Detected - Stopping Program")
                break  

            if left_landmark_list:
                left_distance = get_distance(frame, left_landmark_list)

                # Lock brightness if hand position is stable for 2 seconds
                if prev_left_distance is not None and abs(left_distance - prev_left_distance) < 5:
                    if lock_time is None:
                        lock_time = time.time()  # Start timer
                    elif time.time() - lock_time >= 2:  # If 2 seconds pass
                        brightness_locked = True
                        locked_brightness = int(sbc.get_brightness()[0])  # Get and store current brightness
                else:
                    lock_time = None

                # Unlock brightness using a thumbs-up gesture
                if brightness_locked and detect_thumbs_up(processed):
                    brightness_locked = False

                if not brightness_locked:
                    b_level = np.interp(left_distance, [50, 220], [0, 100])
                    sbc.set_brightness(b_level)

                prev_left_distance = left_distance

         

            # Display lock status with percentage
            if brightness_locked:
                cv2.putText(frame, f"Brightness Locked at {locked_brightness}%", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def get_left_right_landmarks(frame, processed, draw, hands):
    left_landmarks_list = []
    right_landmarks_list = []

    if processed.multi_hand_landmarks:
        for handlm in processed.multi_hand_landmarks:
            for idx, found_landmark in enumerate(handlm.landmark):
                height, width, _ = frame.shape
                x, y = int(found_landmark.x * width), int(found_landmark.y * height)

                if idx in [4, 8]:  # Thumb (4) and Index Finger (8)
                    landmark = [idx, x, y]

                    if handlm == processed.multi_hand_landmarks[0]:
                        left_landmarks_list.append(landmark)
                    elif handlm == processed.multi_hand_landmarks[1]:
                        right_landmarks_list.append(landmark)

            draw.draw_landmarks(frame, handlm, hands.HAND_CONNECTIONS)

    return left_landmarks_list, right_landmarks_list

def get_distance(frame, landmark_list):
    if len(landmark_list) < 2:
        return 0

    (x1, y1), (x2, y2) = (landmark_list[0][1], landmark_list[0][2]), (landmark_list[1][1], landmark_list[1][2])

    cv2.circle(frame, (x1, y1), 7, (0, 255, 0), cv2.FILLED)
    cv2.circle(frame, (x2, y2), 7, (0, 255, 0), cv2.FILLED)
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return hypot(x2 - x1, y2 - y1)

def detect_thumbs_up(processed):
    """Unlock brightness only if a thumbs-up gesture is detected."""
    if not processed.multi_hand_landmarks:
        return False

    for handlm in processed.multi_hand_landmarks:
        # Get the Y positions of thumb (4) and other fingers (index=8, middle=12, ring=16, pinky=20)
        thumb_y = handlm.landmark[4].y
        index_y = handlm.landmark[8].y
        middle_y = handlm.landmark[12].y
        ring_y = handlm.landmark[16].y
        pinky_y = handlm.landmark[20].y

        # Check if thumb is above all other fingers (Thumbs-up sign)
        if thumb_y < index_y and thumb_y < middle_y and thumb_y < ring_y and thumb_y < pinky_y:
            return True

    return False

def detect_smile(frame):
    """Detects a smile using Haar cascades."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)

        if len(smiles) > 0:  # Smile detected
            return True

    return False

if __name__ == "__main__":
    main()
