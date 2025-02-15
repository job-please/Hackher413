import cv2 as cv
# import mss as mss
# from PIL import Image
import numpy as np
import mediapipe as mp
import pyvista as pv
import threading

# mediapipe vision setup
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 3d plot
plot = pv.Plotter()

# create landmarker
with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=.5) as hands:

    # webcam video capture
    vid = cv.VideoCapture(0)
    while True:

        # clear plot
        plot.clear()

        # process frame
        ret, frame = vid.read(0)
        frame = cv.flip(frame, 1)
        mp_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        landmarks = hands.process(mp_image)

        # record pose
        if landmarks.multi_hand_landmarks:

            # get current frame landmark info
            for hand_landmarks in landmarks.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                frame_landmarks = []
                for landmark in hand_landmarks.landmark:
                    coords = [landmark.x,
                              landmark.y,
                              landmark.z]
                    
                    frame_landmarks.append(coords)
                
                frame_landmarks = np.array(frame_landmarks)
                frame_landmarks *= 100
                plot.add_points(frame_landmarks, color='red', point_size=10)
        
        plot.render()

        plot.show()
        
        cv.imshow("VIDEO", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# destroy windows
print("CLOSING")
cv.destroyAllWindows()