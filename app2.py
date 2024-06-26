import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import streamlit as st

# Streamlit setup
st.title("Hand Gesture Drawing App")
st.sidebar.title("Controls")
st.sidebar.markdown("Use the controls below to customize your drawing.")

# Hand gesture controls
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255),
    (255, 165, 0), (128, 0, 128), (255, 20, 147), (0, 191, 255)
]
color_names = ["Blue", "Green", "Red", "Yellow", "Orange", "Purple", "Pink", "Sky Blue"]
colorIndex = st.sidebar.radio("Select Color", list(range(len(colors))), format_func=lambda x: color_names[x])

pen_thickness = st.sidebar.selectbox("Select Pen Thickness", [2, 4, 6, 8, 10, 12, 14, 16], index=2)
eraser_on = st.sidebar.checkbox("Eraser")

# Initialize the canvas
paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

# Initialize deque points for drawing
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
opoints = [deque(maxlen=1024)]
ppoints = [deque(maxlen=1024)]
pkpoints = [deque(maxlen=1024)]
sbpoints = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0
orange_index = 0
purple_index = 0
pink_index = 0
sky_blue_index = 0

# Initialize Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Webcam input
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
while ret:
    # Read frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame
    result = hands.process(framergb)
    
    # Post-process results
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * frame.shape[1])
                lmy = int(lm.y * frame.shape[0])
                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, center, pen_thickness, colors[colorIndex] if not eraser_on else (255, 255, 255), -1)
        
        if thumb[1] - center[1] < 30:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1
            opoints.append(deque(maxlen=512))
            orange_index += 1
            ppoints.append(deque(maxlen=512))
            purple_index += 1
            pkpoints.append(deque(maxlen=512))
            pink_index += 1
            sbpoints.append(deque(maxlen=512))
            sky_blue_index += 1
        elif center[1] <= 65:
            if 40 <= center[0] <= 140:  # Clear Button
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]
                opoints = [deque(maxlen=512)]
                ppoints = [deque(maxlen=512)]
                pkpoints = [deque(maxlen=512)]
                sbpoints = [deque(maxlen=512)]
                blue_index = green_index = red_index = yellow_index = orange_index = purple_index = pink_index = sky_blue_index = 0
                paintWindow[67:, :, :] = 255
            elif 160 <= center[0] <= 255:
                colorIndex = 0  # Blue
            elif 275 <= center[0] <= 370:
                colorIndex = 1  # Green
            elif 390 <= center[0] <= 485:
                colorIndex = 2  # Red
            elif 505 <= center[0] <= 600:
                colorIndex = 3  # Yellow
        else:
            if eraser_on:
                cv2.circle(paintWindow, center, pen_thickness, (255, 255, 255), -1)
            else:
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)
                elif colorIndex == 4:
                    opoints[orange_index].appendleft(center)
                elif colorIndex == 5:
                    ppoints[purple_index].appendleft(center)
                elif colorIndex == 6:
                    pkpoints[pink_index].appendleft(center)
                elif colorIndex == 7:
                    sbpoints[sky_blue_index].appendleft(center)
    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1
        opoints.append(deque(maxlen=512))
        orange_index += 1
        ppoints.append(deque(maxlen=512))
        purple_index += 1
        pkpoints.append(deque(maxlen=512))
        pink_index += 1
        sbpoints.append(deque(maxlen=512))
        sky_blue_index += 1

    points = [bpoints, gpoints, rpoints, ypoints, opoints, ppoints, pkpoints, sbpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], pen_thickness)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], pen_thickness)
    
    # Display the frame and canvas in Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    paintWindowRGB = cv2.cvtColor(paintWindow, cv2.COLOR_BGR2RGB)

    st.image(frame, channels="RGB", use_column_width=True)
    st.image(paintWindowRGB, channels="RGB", use_column_width=True)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
