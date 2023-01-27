import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyautogui as pg

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

model = load_model(r'saved_models\model_600')

# Load class names
f = open('sign_id.txt', 'r')
classNames = f.read().split('\n')
f.close()
#print(classNames)


def gesture_recognition():
    cap = cv2.VideoCapture(1)
    while True:
        # Read each frame from the webcam
        _, frame = cap.read()

        x, y, c = frame.shape # (480, 640, 3)
        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get hand landmark prediction
        result = hands.process(framergb)
        className = ''

        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    # lmx = int(lm.x * x)
                    # lmy = int(lm.y * y)
                    landmarks.append([lm.x, lm.y])


                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                prediction = model.predict(np.reshape(np.ravel(landmarks),(1,42)),verbose=0)
                classID = int(prediction[0]>0.5)
                className = classNames[classID]

                if classID == 0:
                    pg.moveTo(landmarks[8][0]*x*4,landmarks[8][1]*y*1.6875)
                elif classID==1:
                    pg.click()
                    #pg.click(landmarks[8][0]*x*4,landmarks[8][1]*y*1.6875)

        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        # Show the final output
        cv2.imshow("Output", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()


gesture_recognition()