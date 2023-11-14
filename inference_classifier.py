import cv2
import mediapipe as mp
import pickle
import numpy as np
import wordsegment as ws
import time

# MODEL_LOC = 'trained_models/isl_model_2023_10_27_14_54_50.pickle'
MODEL_LOC = 'trained_models/asl_model_2023_11_14_16_48_03.pickle'
ENGLISH_ALPHABETS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

model_dict = pickle.load(open(MODEL_LOC, 'rb'))
# model1 = model_dict['model1']
# model2 = model_dict['model2']
model = model_dict['model']

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# labels_dict = {'A': 'A', 'B': 'B', 'C': 'C'}

char_buffer = ''
ws.load()

while True:

    data_aux = []
    x_ = []
    y_ = []
    x1 = x2 = y1 = y2 = None
    predicted_char = None

    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())
            
        for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)
        if len(x_) != 0:
            x1 = int(min(x_) * W) - 10
            x2 = int(max(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            y2 = int(max(y_) * H) - 10

        # if len(data_aux) == model1.n_features_in_:
        #     prediction = model1.predict([np.asarray(data_aux)])
        #     predicted_char = prediction[0]
        # elif len(data_aux) == model2.n_features_in_:
        #     prediction = model2.predict([np.asarray(data_aux)])
        #     predicted_char = prediction[0]
        if len(data_aux) == model.n_features_in_:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_char = prediction[0]

    if x1 is not None and predicted_char is not None:
        if len(char_buffer) > 30:
            char_buffer = ''
        char_buffer += predicted_char
        text_to_display = ws.segment(char_buffer)
        text_to_display = ' '.join(text_to_display)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
        cv2.rectangle(frame, (10, frame.shape[0]), (frame.shape[1]-10, frame.shape[0]-35), (0,0,0), -1)
        cv2.putText(frame, text_to_display, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        # time.sleep(1)
    cv2.imshow('Frame (Press Q to exit)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()