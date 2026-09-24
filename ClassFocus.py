import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# CLASSFOCUS
# Student Attention Estimation using CV + Supervised ML
# ============================================================

print("=" * 60)
print("                 CLASSFOCUS")
print("=" * 60)
print("Student Attention Detection using CV + ML")
print()
print("Camera starting...")
print("Q = Quit")
print("=" * 60)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# SUPERVISED ML MODEL
# ============================================================

np.random.seed(42)

training_X = []
training_y = []


# ------------------------------------------------------------
# ATTENTIVE SAMPLES
# ------------------------------------------------------------

for _ in range(1000):

    horizontal_movement = np.random.normal(0, 0.025)
    vertical_movement = np.random.normal(0, 0.025)

    movement = np.sqrt(
        horizontal_movement ** 2 +
        vertical_movement ** 2
    )

    horizontal_position = np.random.normal(0, 0.08)
    vertical_position = np.random.normal(0, 0.08)

    training_X.append([
        abs(horizontal_movement),
        abs(vertical_movement),
        movement,
        abs(horizontal_position),
        abs(vertical_position)
    ])

    training_y.append(1)


# ------------------------------------------------------------
# LESS ATTENTIVE SAMPLES
# ------------------------------------------------------------

for _ in range(1000):

    horizontal_movement = np.random.uniform(0.08, 0.35)
    vertical_movement = np.random.uniform(0.08, 0.30)

    movement = np.sqrt(
        horizontal_movement ** 2 +
        vertical_movement ** 2
    )

    horizontal_position = np.random.uniform(0.15, 0.45)
    vertical_position = np.random.uniform(0.15, 0.40)

    training_X.append([
        abs(horizontal_movement),
        abs(vertical_movement),
        movement,
        abs(horizontal_position),
        abs(vertical_position)
    ])

    training_y.append(0)


training_X = np.array(training_X)
training_y = np.array(training_y)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)

model.fit(training_X, training_y)

print("ML model trained successfully.")


# ============================================================
# FACE LANDMARKS
# ============================================================

# Nose
NOSE = 1

# Face left/right landmarks
LEFT_FACE = 234
RIGHT_FACE = 454

# Face top/bottom
TOP_FACE = 10
BOTTOM_FACE = 152


# ============================================================
# TRACK EACH FACE
# ============================================================

face_history = {}

next_face_id = 0


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print()
    print("ERROR: Camera could not be opened.")
    print("Try changing VideoCapture(0) to VideoCapture(1).")
    face_mesh.close()
    exit()


camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# PARAMETERS
# ============================================================

HISTORY_LENGTH = 8

# Movement threshold used to decide when a person
# has moved their head significantly.
MOVEMENT_THRESHOLD = 0.075

# Position threshold for a turned head.
POSITION_THRESHOLD = 0.16

# How long a face can disappear before removing history.
FACE_TIMEOUT = 1.5


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_face_features(landmarks):

    nose = landmarks[NOSE]

    left_face = landmarks[LEFT_FACE]
    right_face = landmarks[RIGHT_FACE]

    top_face = landmarks[TOP_FACE]
    bottom_face = landmarks[BOTTOM_FACE]

    face_width = abs(right_face.x - left_face.x)
    face_height = abs(bottom_face.y - top_face.y)

    if face_width < 0.001:
        face_width = 0.001

    if face_height < 0.001:
        face_height = 0.001

    # Center of face
    face_center_x = (
        left_face.x + right_face.x
    ) / 2

    face_center_y = (
        top_face.y + bottom_face.y
    ) / 2

    # Nose position relative to face center
    relative_x = (
        nose.x - face_center_x
    ) / face_width

    relative_y = (
        nose.y - face_center_y
    ) / face_height

    return (
        relative_x,
        relative_y,
        face_center_x,
        face_center_y,
        face_width,
        face_height
    )


def match_face(current_x, current_y, previous_faces):

    if len(previous_faces) == 0:
        return None

    best_id = None
    best_distance = float("inf")

    for face_id, data in previous_faces.items():

        old_x = data["x"]
        old_y = data["y"]

        distance = np.sqrt(
            (current_x - old_x) ** 2 +
            (current_y - old_y) ** 2
        )

        if distance < best_distance and distance < 0.15:
            best_distance = distance
            best_id = face_id

    return best_id


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print("Could not read camera frame.")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(frame_rgb)


    current_time = time.time()

    total_faces = 0
    attentive_faces = 0


    # ========================================================
    # DETECT FACES
    # ========================================================

    if results.multi_face_landmarks:

        total_faces = len(
            results.multi_face_landmarks
        )


        for face_landmarks in results.multi_face_landmarks:

            landmarks = face_landmarks.landmark


            # ------------------------------------------------
            # FEATURES
            # ------------------------------------------------

            (
                relative_x,
                relative_y,
                face_center_x,
                face_center_y,
                face_width,
                face_height
            ) = calculate_face_features(landmarks)


            # ------------------------------------------------
            # PIXEL COORDINATES
            # ------------------------------------------------

            frame_height, frame_width = frame.shape[:2]

            nose_x = int(
                landmarks[NOSE].x * frame_width
            )

            nose_y = int(
                landmarks[NOSE].y * frame_height
            )

            left_x = int(
                landmarks[LEFT_FACE].x * frame_width
            )

            right_x = int(
                landmarks[RIGHT_FACE].x * frame_width
            )

            top_y = int(
                landmarks[TOP_FACE].y * frame_height
            )

            bottom_y = int(
                landmarks[BOTTOM_FACE].y * frame_height
            )


            # ------------------------------------------------
            # FACE BOX
            # ------------------------------------------------

            x1 = max(0, min(left_x, right_x))
            x2 = min(frame_width - 1, max(left_x, right_x))

            y1 = max(0, min(top_y, bottom_y))
            y2 = min(frame_height - 1, max(top_y, bottom_y))


            # ------------------------------------------------
            # MATCH FACE WITH PREVIOUS FRAME
            # ------------------------------------------------

            matched_id = match_face(
                face_center_x,
                face_center_y,
                face_history
            )


            if matched_id is None:

                matched_id = next_face_id
                next_face_id += 1

                face_history[matched_id] = {
                    "x": face_center_x,
                    "y": face_center_y,
                    "history": deque(
                        maxlen=HISTORY_LENGTH
                    ),
                    "last_seen": current_time
                }


            face_data = face_history[matched_id]


            # ------------------------------------------------
            # CALCULATE HEAD MOVEMENT
            # ------------------------------------------------

            old_x = face_data["x"]
            old_y = face_data["y"]

            movement_x = (
                face_center_x - old_x
            )

            movement_y = (
                face_center_y - old_y
            )

            movement = np.sqrt(
                movement_x ** 2 +
                movement_y ** 2
            )


            # Store movement history
            face_data["history"].append(
                (
                    movement_x,
                    movement_y
                )
            )


            # Update position
            face_data["x"] = face_center_x
            face_data["y"] = face_center_y
            face_data["last_seen"] = current_time


            # ------------------------------------------------
            # AVERAGE MOVEMENT
            # ------------------------------------------------

            if len(face_data["history"]) > 0:

                avg_x = np.mean([
                    abs(x)
                    for x, y
                    in face_data["history"]
                ])

                avg_y = np.mean([
                    abs(y)
                    for x, y
                    in face_data["history"]
                ])

            else:

                avg_x = 0
                avg_y = 0


            avg_movement = np.sqrt(
                avg_x ** 2 +
                avg_y ** 2
            )


            # ------------------------------------------------
            # ML FEATURES
            # ------------------------------------------------

            ml_features = np.array([[
                avg_x,
                avg_y,
                avg_movement,
                abs(relative_x),
                abs(relative_y)
            ]])


            # ------------------------------------------------
            # ML PREDICTION
            # ------------------------------------------------

            prediction = model.predict(
                ml_features
            )[0]

            probability = model.predict_proba(
                ml_features
            )[0]


            confidence = max(probability) * 100


            # ------------------------------------------------
            # EXTRA HEAD MOVEMENT RULE
            # ------------------------------------------------

            # This makes the system more responsive.
            #
            # If the head moves strongly left/right/up/down,
            # classify it as less attentive.

            large_head_movement = (
                avg_movement > MOVEMENT_THRESHOLD
            )

            head_turning = (
                abs(relative_x) > POSITION_THRESHOLD
                or
                abs(relative_y) > POSITION_THRESHOLD
            )


            if large_head_movement or head_turning:

                prediction = 0


            # ------------------------------------------------
            # FINAL STATUS
            # ------------------------------------------------

            if prediction == 1:

                status = "ATTENTIVE"
                attentive_faces += 1

            else:

                status = "LESS ATTENTIVE"


            # ------------------------------------------------
            # DRAW BOX
            # ------------------------------------------------

            if status == "ATTENTIVE":

                box_color = (0, 200, 0)

            else:

                box_color = (0, 0, 255)


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )


            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            label = (
                f"{status} "
                f"{confidence:.0f}%"
            )


            cv2.rectangle(
                frame,
                (x1, max(0, y1 - 35)),
                (
                    min(
                        frame_width,
                        x1 + 240
                    ),
                    y1
                ),
                box_color,
                -1
            )


            cv2.putText(
                frame,
                label,
                (x1 + 5, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


            # ------------------------------------------------
            # NOSE POINT
            # ------------------------------------------------

            cv2.circle(
                frame,
                (nose_x, nose_y),
                4,
                (255, 255, 0),
                -1
            )


    # ========================================================
    # REMOVE OLD FACES
    # ========================================================

    old_faces = []

    for face_id, data in face_history.items():

        if current_time - data["last_seen"] > FACE_TIMEOUT:

            old_faces.append(face_id)


    for face_id in old_faces:

        del face_history[face_id]


    # ========================================================
    # ATTENTION PERCENTAGE
    # ========================================================

    if total_faces > 0:

        attention_percentage = (
            attentive_faces /
            total_faces
        ) * 100

    else:

        attention_percentage = 0


    # ========================================================
    # TOP INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (500, 105),
        (25, 25, 25),
        -1
    )


    cv2.putText(
        frame,
        "CLASSFOCUS",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Class Attention: {attention_percentage:.1f}%",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Students Detected: {total_faces}",
        (20, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    # ========================================================
    # INSTRUCTION
    # ========================================================

    cv2.putText(
        frame,
        "Q = Quit",
        (
            frame.shape[1] - 130,
            frame.shape[0] - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "ClassFocus - Student Attention",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

face_mesh.close()

print()
print("ClassFocus stopped.")