import cv2
from functions import *
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
import numpy as np

model = load_model('./model/lstm.h5')

sequence = []
sentence = []
predictions = []
threshold = 0.9

font = ImageFont.truetype('fonts/SCDream6.otf', 20)

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)
        print(results)

        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print("{} {:.2f}%".format(actions[np.argmax(res)], np.max(res)*100))
            predictions.append(np.argmax(res))

            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:

                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 6:
                sentence.clear()


        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)

        frame = Image.fromarray(image)
        draw = ImageDraw.Draw(frame)
        draw.text(xy=(3, 15), text=" ".join(sentence), font=font, fill=(255, 255, 255))
        image = np.array(frame)

        cv2.imshow('sign_lang', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()