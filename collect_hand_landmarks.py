import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import csv
import pyautogui as pg

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


hand_landmarks_histoty=[]
csv_path='hand_landmarks.csv'


def new_gesture(gesture_id: int):
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
        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    # lmx = int(lm.x * x)
                    # lmy = int(lm.y * y)
                    landmarks.append([lm.x, lm.y])


                hand_landmarks_histoty.append(np.ravel(landmarks))
                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        cv2.putText(frame, f'record gesture_id={gesture_id}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Show the final output
        cv2.imshow("Output", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

    landmarks_df=pd.DataFrame(hand_landmarks_histoty)
    id = pd.DataFrame({'id': np.full(shape=(landmarks_df.shape[0],), fill_value=gesture_id, dtype='int')})
    landmarks_df = pd.concat([id, landmarks_df], axis=1)

    return landmarks_df

def logging_to_csv(hand_landmarks_histoty: pd.DataFrame,csv_path: str):
    hand_landmarks_histoty.to_csv(csv_path,index=False,mode='a',header=False)

temp=new_gesture(1)

#logging_to_csv(temp,csv_path)




