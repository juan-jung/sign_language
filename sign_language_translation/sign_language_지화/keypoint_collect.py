import cv2
import os
import numpy as np
import mediapipe as mp
from functions import *
from PIL import ImageFont, ImageDraw, Image

font = ImageFont.truetype('fonts/SCDream6.otf', 20)

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):

                ret, frame = cap.read()

                image, results = mediapipe_detection(frame, holistic)
                #print(results)

                draw_styled_landmarks(image, results)

                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)

                    frame = Image.fromarray(image)
                    draw = ImageDraw.Draw(frame)
                    draw.text(xy=(3, 15), text="Collecting frames for {} Video Number {}".format(action, sequence), font=font, fill=(0, 255, 0))
                    image = np.array(frame)



                    cv2.imshow('OpenCV', image)
                    cv2.waitKey(3000)
                else:
                    frame = Image.fromarray(image)
                    draw = ImageDraw.Draw(frame)
                    draw.text(xy=(3, 15), text="Collecting frames for {} Video Number {}".format(action, sequence),
                              font=font, fill=(0, 255, 0))
                    image = np.array(frame)
                    cv2.imshow('OpenCV', image)

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()