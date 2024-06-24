import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Global variables for drawing
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

# Colors and other settings
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255),
    (255, 165, 0), (128, 0, 128), (255, 20, 147), (0, 191, 255)
]
color_names = ["Blue", "Green", "Red", "Yellow", "Orange", "Purple", "Pink", "Sky Blue"]
colorIndex = 0
pen_thickness = 5
eraser_on = False
shape_mode = None

# Create a dictionary to store selectable area coordinates and associated actions
selectable_areas = {
    "clear": (40, 1, 140, 65),
    "circle": (700, 1, 770, 65),
    "rectangle": (780, 1, 850, 65),
    "color_blue": (160, 1, 255, 65),
    "color_green": (250, 1, 345, 65),
    "color_red": (340, 1, 435, 65),
    "color_yellow": (430, 1, 525, 65),
    "color_orange": (520, 1, 615, 65),
    "color_purple": (610, 1, 705, 65),
    "color_pink": (700, 1, 795, 65),
    "color_sky_blue": (790, 1, 885, 65),
}

# Here is code for Canvas setup
paintWindow = np.ones((471, 800, 3), dtype=np.uint8) * 255

# Draw interface buttons and selectable areas
def draw_interface():
    global paintWindow
    paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
    cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    for area_name, (x0, y0, x1, y1) in selectable_areas.items():
        paintWindow = cv2.rectangle(paintWindow, (x0, y0), (x1, y1), (150, 150, 150), 2)

    for i, color_name in enumerate(color_names):
        x0, y0, x1, y1 = 160 + i*90, 1, 255 + i*90, 65
        paintWindow = cv2.rectangle(paintWindow, (x0, y0), (x1, y1), colors[i], 2)
        cv2.putText(paintWindow, color_name.upper(), (x0 + 20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(paintWindow, "Thickness:", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    for i, thickness in enumerate([2, 5, 10]):
        paintWindow = cv2.rectangle(paintWindow, (20, 120 + i*40), (120, 160 + i*40), (0,0,0), 2)
        cv2.putText(paintWindow, f"{thickness}px", (50, 145 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

draw_interface()

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True

while ret:
    ret, frame = cap.read()

    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    draw_interface()

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])

        cv2.circle(frame, center, pen_thickness, colors[colorIndex] if not eraser_on else (255, 255, 255), -1)
        
        if (thumb[1] - center[1] < 30):
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
            for area_name, (x0, y0, x1, y1) in selectable_areas.items():
                if x0 <= center[0] <= x1 and y0 <= center[1] <= y1:
                    if area_name == "clear":
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
                    elif area_name.startswith("color_"):
                        colorIndex = list(selectable_areas.keys()).index(area_name)
                        eraser_on = False
                        shape_mode = None
                    elif area_name == "circle":
                        shape_mode = "circle"
                    elif area_name == "rectangle":
                        shape_mode = "rectangle"
                    break

        else:
            if eraser_on:
                cv2.circle(paintWindow, center, pen_thickness, (255, 255, 255), -1)
            elif shape_mode == "circle":
                cv2.circle(paintWindow, center, 30, colors[colorIndex], -1)
                shape_mode = None
            elif shape_mode == "rectangle":
                cv2.rectangle(paintWindow, (center[0] - 30, center[1] - 30), (center[0] + 30, center[1] + 30), colors[colorIndex], -1)
                shape_mode = None
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

    # Draw lines of all the colors on the canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints, opoints, ppoints, pkpoints, sbpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], pen_thickness)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], pen_thickness)

    # Show the output frames
    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    # Keyboard controls for changing pen thickness and eraser
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('e'):
        eraser_on = not eraser_on
        colorIndex = 0  # Switch back to default color after using eraser
    elif key == ord('1'):
        pen_thickness = 2
    elif key == ord('2'):
        pen_thickness = 5
    elif key == ord('3'):
        pen_thickness = 10

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()

