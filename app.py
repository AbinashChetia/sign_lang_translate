from flask import Flask, render_template, Response, redirect, url_for
import cv2
import mediapipe as mp
import pickle
import numpy as np
import wordsegment as ws
import torch
import time

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
MODEL_LOC = 'trained_models/asl_model_2023_11_24_14_18_54.pickle'
model_dict = pickle.load(open(MODEL_LOC, 'rb'))
model = model_dict['model'].to(DEVICE)
lab_encdr = model_dict['lab_encdr']
app = Flask(__name__, template_folder='./templates')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

def start_inferencing():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)

    char_buffer = ''
    text_to_display = ''
    ws.load()

    start_time = time.time()
    delay = 2

    while True:
        data_aux = []
        x_ = []
        y_ = []
        x1 = x2 = y1 = y2 = None
        predicted_char = None

        _, frame = cap.read()
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

            if len(data_aux) == model.n_in_feats:
                prediction = model.predict(torch.from_numpy(np.asarray(data_aux, dtype=np.float32)).unsqueeze(0).to(DEVICE))
                predicted_char = lab_encdr.inverse_transform([prediction.item()])[0]
                if predicted_char == 'nothing' or predicted_char == 'space' or predicted_char == 'del':
                    predicted_char = None

        if x1 is not None and predicted_char is not None and time.time() - start_time > delay:
            start_time = time.time()
            if len(char_buffer) > 30:
                char_buffer = ''
            char_buffer += predicted_char
            text_to_display = ws.segment(char_buffer)
            text_to_display = ' '.join(text_to_display).upper()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
        cv2.rectangle(frame, (0, frame.shape[0]), (frame.shape[1], frame.shape[0]-40), (0,0,0), -1)
        cv2.putText(frame, text_to_display, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        try:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            pass

@app.route('/')
def index():
    return render_template('index.html')
  
@app.route('/infer')
def infer():
    return Response(start_inferencing(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clear')
def clear():
    global char_buffer
    char_buffer = ''
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(port=8000, debug=True)
    # app.run(host='0.0.0.0', port=80, debug=False)

cap.release()
cv2.destroyAllWindows()     