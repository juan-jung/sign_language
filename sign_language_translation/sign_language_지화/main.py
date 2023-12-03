import cv2
from functions import *
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from unicode import join_jamos

model = load_model('./model/cnn.h5')

sequence = []
sentence = []
predictions = []
threshold = 0.8

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
        sequence = sequence[-10:]
        # sequence.insert(0, keypoints)
        # sequence = sequence[:30]

        if len(sequence) == 10:
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


        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.rectangle(image, (0, 440), (640, 480), (145, 117, 16), -1)
        # cv2.putText(image, ' '.join(sentence), (3, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


        if cv2.waitKey(10) & 0xFF == ord('1'):
            sentence.clear()

        if cv2.waitKey(10) & 0xFF == ord('2'):
            sentence.pop()

        s = join_jamos(" ".join(sentence).replace(" ", ""))
        frame1 = Image.fromarray(image)
        draw1 = ImageDraw.Draw(frame1)
        draw1.text(xy=(3, 455), text=s, font=font, fill=(255, 255, 255))
        image = np.array(frame1)

        frame = Image.fromarray(image)
        draw = ImageDraw.Draw(frame)
        draw.text(xy=(3, 15), text=" ".join(sentence), font=font, fill=(255, 255, 255))
        image = np.array(frame)

        cv2.imshow('sign_lang', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()