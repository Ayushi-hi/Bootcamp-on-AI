import cv2
import numpy as np
import os
import urllib.request
import math
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# 1. DOWNLOAD MEDIAPIPE HAND MODEL AUTOMATICALLY
# =========================================================

MODEL_PATH = "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading hand model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Hand model downloaded successfully!")


# =========================================================
# 2. MEDIAPIPE HAND DETECTOR
# =========================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,

    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

detector = vision.HandLandmarker.create_from_options(options)


# =========================================================
# 3. CAMERA
# =========================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    detector.close()
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)


# =========================================================
# 4. CANVAS
# =========================================================

ret, first_frame = cap.read()

if not ret:
    print("ERROR: Could not read camera.")
    cap.release()
    detector.close()
    exit()

first_frame = cv2.flip(first_frame, 1)

height, width = first_frame.shape[:2]

canvas = np.zeros(
    (height, width, 3),
    dtype=np.uint8
)


# =========================================================
# 5. COLORS
# =========================================================

colors = {
    "r": (0, 0, 255),
    "g": (0, 255, 0),
    "b": (255, 0, 0),
    "y": (0, 255, 255),
    "p": (255, 0, 255),
    "w": (255, 255, 255)
}

current_color = colors["r"]
current_color_name = "RED"


# =========================================================
# 6. SMOOTHING VARIABLES
# =========================================================

prev_point = None
smooth_point = None

SMOOTHING = 0.35

MAX_JUMP = 100


# =========================================================
# 7. GESTURE STABILITY
# =========================================================

drawing_state = False

draw_frames = 0
pause_frames = 0

DRAW_CONFIRM_FRAMES = 3
PAUSE_CONFIRM_FRAMES = 3


# =========================================================
# 8. HELPER FUNCTIONS
# =========================================================

def distance(p1, p2):
    """
    Calculate distance between two MediaPipe landmarks.
    """

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def finger_is_extended(hand, tip_id, pip_id, wrist_id=0):
    """
    Check whether a finger is extended.
    """

    tip = hand[tip_id]
    pip = hand[pip_id]
    wrist = hand[wrist_id]

    tip_distance = distance(
        tip,
        wrist
    )

    pip_distance = distance(
        pip,
        wrist
    )

    return tip_distance > pip_distance * 1.10


def is_index_only(hand):
    """
    Draw only when index finger is extended
    and middle, ring and pinky are folded.
    """

    index_extended = finger_is_extended(
        hand,
        8,
        6
    )

    middle_extended = finger_is_extended(
        hand,
        12,
        10
    )

    ring_extended = finger_is_extended(
        hand,
        16,
        14
    )

    pinky_extended = finger_is_extended(
        hand,
        20,
        18
    )

    return (
        index_extended
        and not middle_extended
        and not ring_extended
        and not pinky_extended
    )


def smooth_position(
    old_point,
    new_point,
    amount=0.35
):
    """
    Smooth fingertip movement.
    """

    if old_point is None:
        return new_point

    x = int(
        old_point[0] * (1 - amount)
        + new_point[0] * amount
    )

    y = int(
        old_point[1] * (1 - amount)
        + new_point[1] * amount
    )

    return (x, y)


# =========================================================
# 9. MAIN LOOP
# =========================================================

frame_timestamp = 0

print()
print("======================================")
print("        AIR CANVAS STARTED")
print("======================================")
print("Index finger only = DRAW")
print("Open hand = PAUSE")
print("Fist = PAUSE")
print()
print("R = Red")
print("G = Green")
print("B = Blue")
print("Y = Yellow")
print("P = Pink")
print("W = White")
print("C = Clear")
print("Q = Quit")
print("======================================")


while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Camera frame failed.")
        break


    # =====================================================
    # MIRROR CAMERA
    # =====================================================

    frame = cv2.flip(frame, 1)


    # =====================================================
    # CREATE MEDIAPIPE IMAGE
    # =====================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # IMPORTANT:
    # MediaPipe Tasks API uses mp.Image,
    # NOT vision.Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # =====================================================
    # DETECT HAND
    # =====================================================

    frame_timestamp += 1

    result = detector.detect_for_video(
        mp_image,
        frame_timestamp
    )


    hand_detected = False
    index_only = False


    # =====================================================
    # HAND FOUND
    # =====================================================

    if result.hand_landmarks:

        hand_detected = True

        hand = result.hand_landmarks[0]


        # =================================================
        # CHECK GESTURE
        # =================================================

        index_only = is_index_only(hand)


        # =================================================
        # INDEX FINGER POSITION
        # =================================================

        index_tip = hand[8]

        raw_x = int(
            index_tip.x * width
        )

        raw_y = int(
            index_tip.y * height
        )

        new_point = (
            raw_x,
            raw_y
        )


        # =================================================
        # SMOOTH FINGER POSITION
        # =================================================

        smooth_point = smooth_position(
            smooth_point,
            new_point,
            SMOOTHING
        )


        # =================================================
        # STABILIZE DRAWING STATE
        # =================================================

        if index_only:

            draw_frames += 1
            pause_frames = 0

            if draw_frames >= DRAW_CONFIRM_FRAMES:
                drawing_state = True

        else:

            pause_frames += 1
            draw_frames = 0

            if pause_frames >= PAUSE_CONFIRM_FRAMES:
                drawing_state = False
                prev_point = None


        # =================================================
        # DRAW
        # =================================================

        if drawing_state:

            current_point = smooth_point

            if prev_point is not None:

                dx = (
                    current_point[0]
                    - prev_point[0]
                )

                dy = (
                    current_point[1]
                    - prev_point[1]
                )

                jump_distance = math.sqrt(
                    dx * dx +
                    dy * dy
                )


                # =========================================
                # DRAW ONLY IF MOVEMENT IS REASONABLE
                # =========================================

                if jump_distance < MAX_JUMP:

                    cv2.line(
                        canvas,
                        prev_point,
                        current_point,
                        current_color,
                        6,
                        cv2.LINE_AA
                    )

            prev_point = current_point


        else:

            prev_point = None


        # =================================================
        # DRAW HAND LANDMARKS
        # =================================================

        for landmark in hand:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 255),
                -1
            )


        # =================================================
        # HIGHLIGHT INDEX FINGER
        # =================================================

        cv2.circle(
            frame,
            (raw_x, raw_y),
            10,
            current_color,
            -1
        )


    else:

        # =================================================
        # NO HAND
        # =================================================

        drawing_state = False

        prev_point = None

        smooth_point = None


    # =====================================================
    # COMBINE CAMERA + CANVAS
    # =====================================================

    output = cv2.add(
        frame,
        canvas
    )


    # =====================================================
    # STATUS
    # =====================================================

    if not hand_detected:

        status = "NO HAND"

    elif drawing_state:

        status = "DRAWING"

    else:

        status = "PAUSED - INDEX ONLY"


    # =====================================================
    # STATUS BOX
    # =====================================================

    cv2.rectangle(
        output,
        (10, 10),
        (360, 90),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        status,
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        current_color,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        f"COLOR: {current_color_name}",
        (25, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # =====================================================
    # SHOW WINDOW
    # =====================================================

    cv2.imshow(
        "AI Air Canvas",
        output
    )


    # =====================================================
    # KEYBOARD CONTROLS
    # =====================================================

    key = cv2.waitKey(1) & 0xFF


    # =====================================================
    # QUIT
    # =====================================================

    if key == ord("q"):

        print("Closing Air Canvas...")
        break


    # =====================================================
    # CLEAR
    # =====================================================

    elif key == ord("c"):

        canvas[:] = 0

        prev_point = None

        print("Canvas cleared!")


    # =====================================================
    # COLORS
    # =====================================================

    elif key == ord("r"):

        current_color = colors["r"]
        current_color_name = "RED"

    elif key == ord("g"):

        current_color = colors["g"]
        current_color_name = "GREEN"

    elif key == ord("b"):

        current_color = colors["b"]
        current_color_name = "BLUE"

    elif key == ord("y"):

        current_color = colors["y"]
        current_color_name = "YELLOW"

    elif key == ord("p"):

        current_color = colors["p"]
        current_color_name = "PINK"

    elif key == ord("w"):

        current_color = colors["w"]
        current_color_name = "WHITE"


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

detector.close()

print("Air Canvas closed successfully.")