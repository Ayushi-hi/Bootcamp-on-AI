import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from collections import deque
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# CLASSFOCUS
# Student Attention Detection using CV + Supervised ML
# ============================================================

print("=" * 65)
print("                         CLASSFOCUS")
print("=" * 65)
print("Student Attention Detection using Computer Vision + ML")
print()
print("Camera starting...")
print("Q = Quit and generate session report")
print("=" * 65)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# CREATE SUPERVISED ML DATASET
# ============================================================

np.random.seed(42)

X = []
y = []


# ============================================================
# ATTENTIVE TRAINING DATA
# ============================================================

for _ in range(1000):

    horizontal_movement = np.random.normal(0, 0.02)
    vertical_movement = np.random.normal(0, 0.02)

    movement = np.sqrt(
        horizontal_movement ** 2 +
        vertical_movement ** 2
    )

    horizontal_position = np.random.normal(0, 0.06)
    vertical_position = np.random.normal(0, 0.06)

    X.append([
        abs(horizontal_movement),
        abs(vertical_movement),
        movement,
        abs(horizontal_position),
        abs(vertical_position)
    ])

    y.append(1)


# ============================================================
# LESS ATTENTIVE TRAINING DATA
# ============================================================

for _ in range(1000):

    horizontal_movement = np.random.uniform(0.08, 0.30)
    vertical_movement = np.random.uniform(0.08, 0.25)

    movement = np.sqrt(
        horizontal_movement ** 2 +
        vertical_movement ** 2
    )

    horizontal_position = np.random.uniform(0.18, 0.45)
    vertical_position = np.random.uniform(0.18, 0.40)

    X.append([
        abs(horizontal_movement),
        abs(vertical_movement),
        movement,
        abs(horizontal_position),
        abs(vertical_position)
    ])

    y.append(0)


X = np.array(X, dtype=np.float32)
y = np.array(y)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

print("ML model trained successfully.")


# ============================================================
# FACE LANDMARKS
# ============================================================

NOSE = 1
LEFT_FACE = 234
RIGHT_FACE = 454
TOP_FACE = 10
BOTTOM_FACE = 152


# ============================================================
# FACE TRACKING
# ============================================================

face_history = {}

next_face_id = 0


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():

    camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print()
    print("ERROR: Camera could not be opened.")
    face_mesh.close()
    exit()


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    960
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    540
)


# ============================================================
# SETTINGS
# ============================================================

HISTORY_LENGTH = 6

# Head movement threshold
MOVEMENT_THRESHOLD = 0.035

# Head direction threshold
POSITION_THRESHOLD = 0.16

# Person must remain distracted this long
# before being officially marked distracted.
DISTRACTION_DELAY = 2.0

# Remove a face if it disappears for this long.
FACE_TIMEOUT = 1.0


# ============================================================
# SESSION VARIABLES
# ============================================================

session_start = time.time()

attention_history = []

maximum_students = 0

total_distraction_events = 0


# ============================================================
# FUNCTION: GET FACE FEATURES
# ============================================================

def get_face_features(landmarks):

    nose = landmarks[NOSE]

    left_face = landmarks[LEFT_FACE]
    right_face = landmarks[RIGHT_FACE]

    top_face = landmarks[TOP_FACE]
    bottom_face = landmarks[BOTTOM_FACE]

    face_width = abs(
        right_face.x - left_face.x
    )

    face_height = abs(
        bottom_face.y - top_face.y
    )

    if face_width < 0.001:
        face_width = 0.001

    if face_height < 0.001:
        face_height = 0.001

    face_center_x = (
        left_face.x +
        right_face.x
    ) / 2

    face_center_y = (
        top_face.y +
        bottom_face.y
    ) / 2

    relative_x = (
        nose.x -
        face_center_x
    ) / face_width

    relative_y = (
        nose.y -
        face_center_y
    ) / face_height

    return (
        relative_x,
        relative_y,
        face_center_x,
        face_center_y
    )


# ============================================================
# FUNCTION: FIND EXISTING FACE
# ============================================================

def find_face(current_x, current_y):

    best_id = None

    best_distance = 999

    for face_id, data in face_history.items():

        distance = np.sqrt(
            (current_x - data["x"]) ** 2 +
            (current_y - data["y"]) ** 2
        )

        if (
            distance < best_distance
            and distance < 0.15
        ):

            best_distance = distance
            best_id = face_id

    return best_id


# ============================================================
# FUNCTION: FORMAT TIME
# ============================================================

def format_duration(seconds):

    seconds = int(seconds)

    minutes = seconds // 60

    remaining_seconds = seconds % 60

    return f"{minutes:02d}:{remaining_seconds:02d}"


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print("Camera frame could not be read.")

        break


    # --------------------------------------------------------
    # MIRROR CAMERA
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)


    # --------------------------------------------------------
    # CONVERT TO RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    results = face_mesh.process(rgb)


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


        # Update maximum number of students
        if total_faces > maximum_students:

            maximum_students = total_faces


        # ====================================================
        # PROCESS EACH FACE
        # ====================================================

        for face_landmarks in results.multi_face_landmarks:

            landmarks = face_landmarks.landmark


            # ------------------------------------------------
            # GET FEATURES
            # ------------------------------------------------

            (
                relative_x,
                relative_y,
                face_center_x,
                face_center_y
            ) = get_face_features(
                landmarks
            )


            # ------------------------------------------------
            # FIND FACE ID
            # ------------------------------------------------

            face_id = find_face(
                face_center_x,
                face_center_y
            )


            # ------------------------------------------------
            # CREATE NEW FACE
            # ------------------------------------------------

            if face_id is None:

                face_id = next_face_id

                next_face_id += 1

                face_history[face_id] = {

                    "x": face_center_x,

                    "y": face_center_y,

                    "history": deque(
                        maxlen=HISTORY_LENGTH
                    ),

                    "last_seen": current_time,

                    # Distraction timer
                    "distraction_start": None,

                    # Whether current distraction
                    # has already been counted
                    "distraction_counted": False
                }


            data = face_history[face_id]


            # ------------------------------------------------
            # HEAD MOVEMENT
            # ------------------------------------------------

            movement_x = (
                face_center_x -
                data["x"]
            )

            movement_y = (
                face_center_y -
                data["y"]
            )


            movement = np.sqrt(
                movement_x ** 2 +
                movement_y ** 2
            )


            # Store movement

            data["history"].append(
                (
                    movement_x,
                    movement_y
                )
            )


            # Update position

            data["x"] = face_center_x

            data["y"] = face_center_y

            data["last_seen"] = current_time


            # ------------------------------------------------
            # AVERAGE MOVEMENT
            # ------------------------------------------------

            if len(data["history"]) > 0:

                avg_x = np.mean([
                    abs(item[0])
                    for item in data["history"]
                ])

                avg_y = np.mean([
                    abs(item[1])
                    for item in data["history"]
                ])

            else:

                avg_x = 0

                avg_y = 0


            avg_movement = np.sqrt(
                avg_x ** 2 +
                avg_y ** 2
            )


            # =================================================
            # ML FEATURES
            # =================================================

            features = np.array(
                [[
                    float(avg_x),
                    float(avg_y),
                    float(avg_movement),
                    float(abs(relative_x)),
                    float(abs(relative_y))
                ]],
                dtype=np.float32
            )


            # =================================================
            # ML PREDICTION
            # =================================================

            prediction = int(
                model.predict(features)[0]
            )


            probabilities = model.predict_proba(
                features
            )[0]


            confidence = float(
                max(probabilities) * 100
            )


            # =================================================
            # HEAD MOVEMENT / POSITION
            # =================================================

            strong_movement = (
                avg_movement >
                MOVEMENT_THRESHOLD
            )


            head_turn = (

                abs(relative_x) >
                POSITION_THRESHOLD

                or

                abs(relative_y) >
                POSITION_THRESHOLD
            )


            # =================================================
            # INITIAL DISTRACTION STATE
            # =================================================

            potentially_distracted = (

                prediction == 0

                or

                strong_movement

                or

                head_turn
            )


            # =================================================
            # DISTRACTION TIMER
            # =================================================

            if potentially_distracted:

                # Start timer
                if data["distraction_start"] is None:

                    data["distraction_start"] = (
                        current_time
                    )

                    data["distraction_counted"] = False


                distraction_duration = (
                    current_time -
                    data["distraction_start"]
                )


                # --------------------------------------------
                # ONLY MARK DISTRACTED AFTER 2 SECONDS
                # --------------------------------------------

                if distraction_duration >= DISTRACTION_DELAY:

                    status = "DISTRACTED"

                    box_color = (0, 0, 255)


                    # Count event only once
                    if not data["distraction_counted"]:

                        total_distraction_events += 1

                        data["distraction_counted"] = True


                else:

                    # Still inside 2-second grace period
                    status = "CHECKING..."

                    box_color = (0, 165, 255)


            else:

                # Person is attentive
                status = "ATTENTIVE"

                box_color = (0, 200, 0)


                # Reset distraction timer

                data["distraction_start"] = None

                data["distraction_counted"] = False


            # =================================================
            # ATTENTION COUNT
            # =================================================

            if status == "ATTENTIVE":

                attentive_faces += 1


            # =================================================
            # FACE BOX
            # =================================================

            frame_h, frame_w = frame.shape[:2]


            x_values = [

                int(
                    landmarks[234].x *
                    frame_w
                ),

                int(
                    landmarks[454].x *
                    frame_w
                )
            ]


            y_values = [

                int(
                    landmarks[10].y *
                    frame_h
                ),

                int(
                    landmarks[152].y *
                    frame_h
                )
            ]


            x1 = max(
                0,
                min(x_values)
            )


            x2 = min(
                frame_w - 1,
                max(x_values)
            )


            y1 = max(
                0,
                min(y_values)
            )


            y2 = min(
                frame_h - 1,
                max(y_values)
            )


            # =================================================
            # DRAW FACE BOX
            # =================================================

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                box_color,

                2
            )


            # =================================================
            # STATUS LABEL
            # =================================================

            cv2.rectangle(

                frame,

                (
                    x1,
                    max(
                        0,
                        y1 - 32
                    )
                ),

                (
                    min(
                        frame_w,
                        x1 + 230
                    ),

                    y1
                ),

                box_color,

                -1
            )


            cv2.putText(

                frame,

                status,

                (
                    x1 + 5,
                    max(
                        20,
                        y1 - 10
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (255, 255, 255),

                2
            )


            # =================================================
            # DISTRACTION DURATION
            # =================================================

            if (
                data["distraction_start"]
                is not None
            ):

                current_distraction = (

                    current_time -
                    data["distraction_start"]
                )


                if current_distraction > 0:

                    duration_text = (
                        f"{current_distraction:.1f}s"
                    )


                    cv2.putText(

                        frame,

                        f"Distraction: {duration_text}",

                        (
                            x1,
                            min(
                                frame_h - 35,
                                y2 + 22
                            )
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.48,

                        box_color,

                        2
                    )


            # =================================================
            # NOSE POINT
            # =================================================

            nose_x = int(
                landmarks[NOSE].x *
                frame_w
            )


            nose_y = int(
                landmarks[NOSE].y *
                frame_h
            )


            cv2.circle(

                frame,

                (
                    nose_x,
                    nose_y
                ),

                4,

                (255, 255, 0),

                -1
            )


            # =================================================
            # HEAD DIRECTION
            # =================================================

            if abs(relative_x) > POSITION_THRESHOLD:

                if relative_x > 0:

                    direction = "RIGHT"

                else:

                    direction = "LEFT"


            elif abs(relative_y) > POSITION_THRESHOLD:

                if relative_y > 0:

                    direction = "DOWN"

                else:

                    direction = "UP"


            else:

                if strong_movement:

                    direction = "MOVING"

                else:

                    direction = "CENTER"


            cv2.putText(

                frame,

                f"Head: {direction}",

                (
                    x1,
                    min(
                        frame_h - 10,
                        y2 + 42
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                box_color,

                2
            )


    # ========================================================
    # REMOVE OLD FACES
    # ========================================================

    remove_ids = []


    for face_id, data in face_history.items():

        if (

            current_time -
            data["last_seen"]

            >

            FACE_TIMEOUT

        ):

            remove_ids.append(face_id)


    for face_id in remove_ids:

        del face_history[face_id]


    # ========================================================
    # CLASS ATTENTION
    # ========================================================

    if total_faces > 0:

        attention_percentage = (

            attentive_faces /
            total_faces

        ) * 100

    else:

        attention_percentage = 0


    # Store attention for report

    attention_history.append(
        attention_percentage
    )


    # ========================================================
    # SESSION STATISTICS
    # ========================================================

    session_duration = (
        current_time -
        session_start
    )


    if len(attention_history) > 0:

        average_attention = np.mean(
            attention_history
        )

        peak_attention = np.max(
            attention_history
        )

        lowest_attention = np.min(
            attention_history
        )

    else:

        average_attention = 0

        peak_attention = 0

        lowest_attention = 0


    # ========================================================
    # TOP INFORMATION PANEL
    # ========================================================

    cv2.rectangle(

        frame,

        (0, 0),

        (500, 145),

        (25, 25, 25),

        -1
    )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(

        frame,

        "CLASSFOCUS",

        (20, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2
    )


    # ========================================================
    # ATTENTION
    # ========================================================

    cv2.putText(

        frame,

        f"Class Attention: {attention_percentage:.1f}%",

        (20, 62),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (0, 255, 0),

        2
    )


    # ========================================================
    # STUDENTS
    # ========================================================

    cv2.putText(

        frame,

        f"Students: {total_faces}",

        (20, 88),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        1
    )


    # ========================================================
    # DISTRACTION EVENTS
    # ========================================================

    cv2.putText(

        frame,

        f"Distractions: {total_distraction_events}",

        (20, 112),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 165, 255),

        1
    )


    # ========================================================
    # SESSION TIME
    # ========================================================

    cv2.putText(

        frame,

        f"Session: {format_duration(session_duration)}",

        (20, 137),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255, 255, 255),

        1
    )


    # ========================================================
    # QUIT
    # ========================================================

    cv2.putText(

        frame,

        "Q = Save Report & Quit",

        (
            frame.shape[1] - 200,
            frame.shape[0] - 20
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255, 255, 255),

        1
    )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(

        "ClassFocus - Student Attention",

        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q") or key == ord("Q"):

        break


# ============================================================
# FINAL SESSION TIME
# ============================================================

session_end = time.time()

final_session_duration = (
    session_end -
    session_start
)


# ============================================================
# FINAL STATISTICS
# ============================================================

if len(attention_history) > 0:

    final_average_attention = np.mean(
        attention_history
    )

    final_peak_attention = np.max(
        attention_history
    )

    final_lowest_attention = np.min(
        attention_history
    )

else:

    final_average_attention = 0

    final_peak_attention = 0

    final_lowest_attention = 0


# ============================================================
# SAVE CSV REPORT
# ============================================================

report_file = "classfocus_report.csv"


with open(
    report_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)


    writer.writerow([
        "Metric",
        "Value"
    ])


    writer.writerow([
        "Maximum Students Detected",
        maximum_students
    ])


    writer.writerow([
        "Average Attention",
        f"{final_average_attention:.2f}%"
    ])


    writer.writerow([
        "Peak Attention",
        f"{final_peak_attention:.2f}%"
    ])


    writer.writerow([
        "Lowest Attention",
        f"{final_lowest_attention:.2f}%"
    ])


    writer.writerow([
        "Session Duration",
        format_duration(
            final_session_duration
        )
    ])


    writer.writerow([
        "Distraction Events",
        total_distraction_events
    ])


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

face_mesh.close()


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print()
print("=" * 65)
print("                 SESSION COMPLETE")
print("=" * 65)

print(
    f"Maximum Students Detected : {maximum_students}"
)

print(
    f"Average Attention         : "
    f"{final_average_attention:.2f}%"
)

print(
    f"Peak Attention            : "
    f"{final_peak_attention:.2f}%"
)

print(
    f"Lowest Attention          : "
    f"{final_lowest_attention:.2f}%"
)

print(
    f"Session Duration          : "
    f"{format_duration(final_session_duration)}"
)

print(
    f"Distraction Events        : "
    f"{total_distraction_events}"
)

print()
print(
    f"Report saved as: {report_file}"
)

print("=" * 65)