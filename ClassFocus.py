import os

# Reduce TensorFlow/MediaPipe console messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import time
import csv

from collections import deque
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# CONFIGURATION
# ============================================================

MAX_FACES = 5

HISTORY_LENGTH = 6

MOVEMENT_THRESHOLD = 0.035
POSITION_THRESHOLD = 0.16

# Student must remain potentially distracted
# for this many seconds before being marked distracted.
DISTRACTION_DELAY = 2.0

# Lower value = stricter recognition.
RECOGNITION_THRESHOLD = 0.55

CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540

# How close two face centers must be to be considered
# the same tracked face.
FACE_MATCH_THRESHOLD = 0.15


# ============================================================
# FILES
# ============================================================

STUDENT_FILE = "classfocus_students.npz"
REPORT_FILE = "classfocus_report.csv"


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=MAX_FACES,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# FACE LANDMARK INDEXES
# ============================================================

NOSE = 1

LEFT_FACE = 234
RIGHT_FACE = 454

TOP_FACE = 10
BOTTOM_FACE = 152


# ============================================================
# STUDENT DATABASE
# ============================================================

student_names = []
student_signatures = []


# ============================================================
# LOAD STUDENTS
# ============================================================

def load_students():

    global student_names
    global student_signatures

    if not os.path.exists(STUDENT_FILE):
        print("No enrolled student database found.")
        return

    try:

        data = np.load(
            STUDENT_FILE,
            allow_pickle=True
        )

        student_names = list(data["names"])

        signatures = data["signatures"]

        student_signatures = [
            np.array(signature, dtype=np.float32)
            for signature in signatures
        ]

        print()
        print("========================================")
        print("ENROLLED STUDENTS")
        print("========================================")

        for name in student_names:
            print("-", name)

        print()

    except Exception as e:

        print("Could not load student database.")
        print("Error:", e)

        student_names = []
        student_signatures = []


# ============================================================
# SAVE STUDENTS
# ============================================================

def save_students():

    if len(student_names) == 0:
        return

    try:

        np.savez(
            STUDENT_FILE,
            names=np.array(
                student_names,
                dtype=object
            ),
            signatures=np.array(
                student_signatures,
                dtype=object
            )
        )

        print("Student database saved.")

    except Exception as e:

        print("Could not save student database.")
        print("Error:", e)


# ============================================================
# CREATE FACE SIGNATURE
# ============================================================

def create_face_signature(face_landmarks):

    points = []

    for landmark in face_landmarks.landmark:

        points.append([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    points = np.array(
        points,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # FACE BOUNDING BOX
    # --------------------------------------------------------

    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])

    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    width = max_x - min_x
    height = max_y - min_y

    if width < 0.0001:
        width = 0.0001

    if height < 0.0001:
        height = 0.0001

    # --------------------------------------------------------
    # NORMALIZE X/Y
    # --------------------------------------------------------

    points[:, 0] = (
        points[:, 0] - min_x
    ) / width

    points[:, 1] = (
        points[:, 1] - min_y
    ) / height

    # --------------------------------------------------------
    # NORMALIZE Z
    # --------------------------------------------------------

    points[:, 2] = points[:, 2] / width

    # --------------------------------------------------------
    # FLATTEN
    # --------------------------------------------------------

    signature = points.flatten()

    # --------------------------------------------------------
    # VECTOR NORMALIZATION
    # --------------------------------------------------------

    norm = np.linalg.norm(signature)

    if norm > 0:

        signature = (
            signature / norm
        )

    return signature.astype(
        np.float32
    )


# ============================================================
# FACE RECOGNITION
# ============================================================

def recognize_student(signature):

    if len(student_names) == 0:

        return "Unknown", None

    best_name = "Unknown"
    best_distance = float("inf")

    for name, stored_signature in zip(
        student_names,
        student_signatures
    ):

        if len(stored_signature) != len(signature):
            continue

        distance = np.linalg.norm(
            signature - stored_signature
        )

        if distance < best_distance:

            best_distance = distance
            best_name = name

    if best_distance <= RECOGNITION_THRESHOLD:

        return (
            best_name,
            best_distance
        )

    return (
        "Unknown",
        best_distance
    )


# ============================================================
# ENROLL STUDENT
# ============================================================

def enroll_student(cap):

    print()
    print("========================================")
    print("STUDENT ENROLLMENT")
    print("========================================")

    name = input(
        "Enter student name: "
    ).strip()

    if not name:

        print("Invalid name.")
        return

    # --------------------------------------------------------
    # CHECK DUPLICATE
    # --------------------------------------------------------

    if name.lower() in [
        n.lower()
        for n in student_names
    ]:

        print(
            "This student is already enrolled."
        )

        return

    print()
    print("Look directly at the camera.")
    print("Move your face slightly.")
    print("Capturing face samples...")
    print("Press Q to cancel.")
    print()

    samples = []

    capture_start = time.time()

    while len(samples) < 20:

        ret, frame = cap.read()

        if not ret:

            print(
                "Could not read camera."
            )

            return

        frame = cv2.flip(
            frame,
            1
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(
            rgb
        )

        display = frame.copy()

        # ----------------------------------------------------
        # ENROLLMENT TEXT
        # ----------------------------------------------------

        cv2.putText(
            display,
            f"Enrolling: {name}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Samples: {len(samples)}/20",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # FACE DETECTED
        # ----------------------------------------------------

        if results.multi_face_landmarks:

            face = (
                results.multi_face_landmarks[0]
            )

            signature = (
                create_face_signature(face)
            )

            samples.append(signature)

            # ------------------------------------------------
            # FACE BOX
            # ------------------------------------------------

            xs = [
                landmark.x
                for landmark in face.landmark
            ]

            ys = [
                landmark.y
                for landmark in face.landmark
            ]

            h, w = display.shape[:2]

            x1 = int(
                min(xs) * w
            )

            y1 = int(
                min(ys) * h
            )

            x2 = int(
                max(xs) * w
            )

            y2 = int(
                max(ys) * h
            )

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(
                w - 1,
                x2
            )

            y2 = min(
                h - 1,
                y2
            )

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                "FACE DETECTED",
                (
                    x1,
                    max(
                        25,
                        y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                display,
                "NO FACE DETECTED",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "ClassFocus - Student Enrollment",
            display
        )

        key = cv2.waitKey(100) & 0xFF

        if key == ord("q"):

            print(
                "Enrollment cancelled."
            )

            cv2.destroyWindow(
                "ClassFocus - Student Enrollment"
            )

            return

        # ----------------------------------------------------
        # TIMEOUT
        # ----------------------------------------------------

        if (
            time.time()
            - capture_start
            > 30
        ):

            print(
                "Enrollment timed out."
            )

            cv2.destroyWindow(
                "ClassFocus - Student Enrollment"
            )

            return

    # ========================================================
    # CREATE AVERAGE SIGNATURE
    # ========================================================

    average_signature = np.mean(
        np.array(samples),
        axis=0
    )

    norm = np.linalg.norm(
        average_signature
    )

    if norm > 0:

        average_signature = (
            average_signature / norm
        )

    student_names.append(name)

    student_signatures.append(
        average_signature.astype(
            np.float32
        )
    )

    save_students()

    cv2.destroyWindow(
        "ClassFocus - Student Enrollment"
    )

    print()
    print(
        "Student enrolled successfully!"
    )

    print(
        "Name:",
        name
    )

    print(
        "Total enrolled students:",
        len(student_names)
    )

    print()


# ============================================================
# TRAIN SUPERVISED ML MODEL
# ============================================================

def train_attention_model():

    print(
        "Creating supervised training dataset..."
    )

    rng = np.random.default_rng(42)

    # ========================================================
    # ATTENTIVE DATA
    # ========================================================

    attentive = np.column_stack([

        # Horizontal position
        rng.normal(
            0,
            0.02,
            300
        ),

        # Vertical position
        rng.normal(
            0,
            0.02,
            300
        ),

        # Head movement
        rng.normal(
            0.015,
            0.008,
            300
        ),

        # Horizontal distance
        rng.normal(
            0.03,
            0.015,
            300
        ),

        # Vertical distance
        rng.normal(
            0.03,
            0.015,
            300
        )
    ])

    # ========================================================
    # LESS ATTENTIVE DATA
    # ========================================================

    distracted = np.column_stack([

        # Horizontal position
        rng.normal(
            0,
            0.10,
            300
        ),

        # Vertical position
        rng.normal(
            0,
            0.10,
            300
        ),

        # Head movement
        rng.normal(
            0.08,
            0.025,
            300
        ),

        # Horizontal distance
        rng.normal(
            0.20,
            0.07,
            300
        ),

        # Vertical distance
        rng.normal(
            0.20,
            0.07,
            300
        )
    ])

    # ========================================================
    # COMBINE DATA
    # ========================================================

    X = np.vstack([
        attentive,
        distracted
    ])

    y = np.array(
        [1] * len(attentive)
        +
        [0] * len(distracted)
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=6,
        random_state=42,
        n_jobs=1
    )

    model.fit(
        X,
        y
    )

    print(
        "Attention model ready."
    )

    return model


# ============================================================
# GET FACE INFORMATION
# ============================================================

def get_face_information(
    face_landmarks
):

    nose = (
        face_landmarks.landmark[NOSE]
    )

    left = (
        face_landmarks.landmark[
            LEFT_FACE
        ]
    )

    right = (
        face_landmarks.landmark[
            RIGHT_FACE
        ]
    )

    top = (
        face_landmarks.landmark[
            TOP_FACE
        ]
    )

    bottom = (
        face_landmarks.landmark[
            BOTTOM_FACE
        ]
    )

    center_x = (
        left.x
        +
        right.x
    ) / 2

    center_y = (
        top.y
        +
        bottom.y
    ) / 2

    face_width = abs(
        right.x
        -
        left.x
    )

    if face_width < 0.001:

        face_width = 0.001

    relative_x = (
        nose.x
        -
        center_x
    ) / face_width

    relative_y = (
        nose.y
        -
        center_y
    ) / face_width

    return (
        nose.x,
        nose.y,
        relative_x,
        relative_y
    )


# ============================================================
# DISTANCE BETWEEN TWO FACE CENTERS
# ============================================================

def calculate_distance(
    point1,
    point2
):

    return np.sqrt(
        (
            point1[0]
            -
            point2[0]
        ) ** 2
        +
        (
            point1[1]
            -
            point2[1]
        ) ** 2
    )


# ============================================================
# CREATE ML FEATURES
# ============================================================

def create_features(
    avg_x,
    avg_y,
    avg_movement,
    relative_x,
    relative_y
):

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

    return features


# ============================================================
# STUDENT STATISTICS
# ============================================================

student_stats = {}


def initialize_student(name):

    if name not in student_stats:

        student_stats[name] = {

            "frames": 0,

            "attentive_frames": 0,

            "distracted_frames": 0,

            "distraction_events": 0,

            "max_distraction": 0.0
        }


# ============================================================
# UPDATE STUDENT STATISTICS
# ============================================================

def update_student_stats(
    name,
    is_attentive,
    distraction_duration
):

    initialize_student(name)

    stats = student_stats[name]

    stats["frames"] += 1

    if is_attentive:

        stats[
            "attentive_frames"
        ] += 1

    else:

        stats[
            "distracted_frames"
        ] += 1

    if (
        distraction_duration
        >
        stats["max_distraction"]
    ):

        stats[
            "max_distraction"
        ] = distraction_duration


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    session_duration,
    maximum_students,
    class_attention_history,
    total_distraction_events
):

    # ========================================================
    # CLASS STATISTICS
    # ========================================================

    if len(
        class_attention_history
    ) > 0:

        average_attention = (
            sum(
                class_attention_history
            )
            /
            len(
                class_attention_history
            )
        )

        peak_attention = max(
            class_attention_history
        )

        lowest_attention = min(
            class_attention_history
        )

    else:

        average_attention = 0

        peak_attention = 0

        lowest_attention = 0

    rows = []

    # ========================================================
    # CLASS SUMMARY
    # ========================================================

    rows.append([
        "CLASS SUMMARY",
        "",
        "",
        "",
        "",
        ""
    ])

    rows.append([
        "Maximum Students",
        maximum_students,
        "",
        "",
        "",
        ""
    ])

    rows.append([
        "Average Attention",
        f"{average_attention:.2f}%",
        "",
        "",
        "",
        ""
    ])

    rows.append([
        "Peak Attention",
        f"{peak_attention:.2f}%",
        "",
        "",
        "",
        ""
    ])

    rows.append([
        "Lowest Attention",
        f"{lowest_attention:.2f}%",
        "",
        "",
        "",
        ""
    ])

    rows.append([
        "Session Duration",
        f"{session_duration:.1f} seconds",
        "",
        "",
        "",
        ""
    ])

    rows.append([
        "Total Distraction Events",
        total_distraction_events,
        "",
        "",
        "",
        ""
    ])

    rows.append([
        "",
        "",
        "",
        "",
        "",
        ""
    ])

    # ========================================================
    # INDIVIDUAL REPORT
    # ========================================================

    rows.append([
        "INDIVIDUAL STUDENT REPORT",
        "",
        "",
        "",
        "",
        ""
    ])

    rows.append([
        "Student",
        "Attention %",
        "Attentive Frames",
        "Distracted Frames",
        "Distraction Events",
        "Max Distraction (sec)"
    ])

    for name, stats in student_stats.items():

        total_frames = (
            stats["frames"]
        )

        if total_frames > 0:

            attention_percentage = (
                stats[
                    "attentive_frames"
                ]
                /
                total_frames
            ) * 100

        else:

            attention_percentage = 0

        rows.append([
            name,
            f"{attention_percentage:.2f}%",
            stats[
                "attentive_frames"
            ],
            stats[
                "distracted_frames"
            ],
            stats[
                "distraction_events"
            ],
            f"{stats['max_distraction']:.2f}"
        ])

    # ========================================================
    # WRITE CSV
    # ========================================================

    try:

        with open(
            REPORT_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerows(rows)

        print()
        print(
            "========================================"
        )

        print(
            "REPORT SAVED"
        )

        print(
            "========================================"
        )

        print(
            "File:",
            REPORT_FILE
        )

    except Exception as e:

        print(
            "Could not save report."
        )

        print(
            "Error:",
            e
        )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    global student_stats

    print()
    print(
        "=============================================="
    )

    print(
        "              CLASSFOCUS"
    )

    print(
        "     Classroom Attention Detection"
    )

    print(
        "=============================================="
    )

    print()

    # ========================================================
    # LOAD STUDENTS
    # ========================================================

    load_students()

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    print(
        "Training attention model..."
    )

    model = train_attention_model()

    # ========================================================
    # CAMERA
    # ========================================================

    print(
        "Starting camera..."
    )

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        print(
            "DirectShow camera failed."
        )

        print(
            "Trying default camera..."
        )

        cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print()
        print(
            "ERROR: Could not open camera."
        )

        print(
            "Check whether another application "
            "is using your camera."
        )

        return

    # ========================================================
    # CAMERA RESOLUTION
    # ========================================================

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    # ========================================================
    # FACE TRACKING
    # ========================================================

    face_history = {}

    next_face_id = 0

    # ========================================================
    # SESSION VARIABLES
    # ========================================================

    session_start = time.time()

    attention_history = []

    maximum_students = 0

    total_distraction_events = 0

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "Could not read camera frame."
            )

            break

        # ----------------------------------------------------
        # MIRROR CAMERA
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        frame_height, frame_width = (
            frame.shape[:2]
        )

        # ----------------------------------------------------
        # MEDIAPIPE
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(
            rgb_frame
        )

        current_faces = []

        # ====================================================
        # DETECT FACES
        # ====================================================

        if results.multi_face_landmarks:

            for face_landmarks in (
                results.multi_face_landmarks
            ):

                # ------------------------------------------------
                # FACE POSITION
                # ------------------------------------------------

                (
                    nose_x,
                    nose_y,
                    relative_x,
                    relative_y
                ) = get_face_information(
                    face_landmarks
                )

                center = (
                    nose_x,
                    nose_y
                )

                # ------------------------------------------------
                # FACE SIGNATURE
                # ------------------------------------------------

                signature = (
                    create_face_signature(
                        face_landmarks
                    )
                )

                # ------------------------------------------------
                # RECOGNITION
                # ------------------------------------------------

                (
                    name,
                    recognition_distance
                ) = recognize_student(
                    signature
                )

                current_faces.append({

                    "landmarks":
                        face_landmarks,

                    "center":
                        center,

                    "relative_x":
                        relative_x,

                    "relative_y":
                        relative_y,

                    "signature":
                        signature,

                    "name":
                        name,

                    "recognition_distance":
                        recognition_distance
                })

        # ====================================================
        # FACE TRACKING
        # ====================================================

        matched_ids = set()

        for face in current_faces:

            best_id = None

            best_distance = float(
                "inf"
            )

            # ------------------------------------------------
            # FIND CLOSEST PREVIOUS FACE
            # ------------------------------------------------

            for face_id, data in (
                face_history.items()
            ):

                if face_id in matched_ids:

                    continue

                distance = calculate_distance(
                    face["center"],
                    data["center"]
                )

                if (
                    distance
                    <
                    best_distance
                ):

                    best_distance = (
                        distance
                    )

                    best_id = face_id

            # ------------------------------------------------
            # CREATE NEW FACE TRACK
            # ------------------------------------------------

            if (
                best_id is None
                or
                best_distance
                >
                FACE_MATCH_THRESHOLD
            ):

                best_id = next_face_id

                next_face_id += 1

                face_history[best_id] = {

                    "center":
                        face["center"],

                    "history":
                        deque(
                            maxlen=HISTORY_LENGTH
                        ),

                    "distraction_start":
                        None,

                    "distraction_counted":
                        False,

                    "name":
                        face["name"]
                }

            matched_ids.add(
                best_id
            )

            data = (
                face_history[best_id]
            )

            # ------------------------------------------------
            # UPDATE RECOGNIZED NAME
            # ------------------------------------------------

            if (
                face["name"]
                !=
                "Unknown"
            ):

                data["name"] = (
                    face["name"]
                )

            name = data["name"]

            # ------------------------------------------------
            # MOVEMENT
            # ------------------------------------------------

            previous_center = (
                data["center"]
            )

            movement = calculate_distance(
                face["center"],
                previous_center
            )

            data["center"] = (
                face["center"]
            )

            data["history"].append(
                movement
            )

            if len(
                data["history"]
            ) > 0:

                avg_movement = (
                    sum(
                        data["history"]
                    )
                    /
                    len(
                        data["history"]
                    )
                )

            else:

                avg_movement = 0.0

            # ------------------------------------------------
            # POSITION
            # ------------------------------------------------

            avg_x = (
                face["center"][0]
                -
                0.5
            )

            avg_y = (
                face["center"][1]
                -
                0.5
            )

            relative_x = (
                face["relative_x"]
            )

            relative_y = (
                face["relative_y"]
            )

            # ------------------------------------------------
            # ML FEATURES
            # ------------------------------------------------

            features = create_features(
                avg_x,
                avg_y,
                avg_movement,
                relative_x,
                relative_y
            )

            # =================================================
            # ML PREDICTION
            # =================================================

            try:

                prediction = int(
                    model.predict(
                        features
                    )[0]
                )

                probabilities = (
                    model.predict_proba(
                        features
                    )[0]
                )

                confidence = float(
                    max(probabilities)
                    * 100
                )

            except Exception as e:

                print(
                    "Prediction error:",
                    e
                )

                prediction = 1

                confidence = 0.0

            # =================================================
            # HEAD MOVEMENT RULE
            # =================================================

            strong_movement = (
                avg_movement
                >
                MOVEMENT_THRESHOLD
            )

            head_turn = (
                abs(relative_x)
                >
                POSITION_THRESHOLD
                or
                abs(relative_y)
                >
                POSITION_THRESHOLD
            )

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

            distraction_duration = 0.0

            if potentially_distracted:

                if (
                    data[
                        "distraction_start"
                    ]
                    is None
                ):

                    data[
                        "distraction_start"
                    ] = time.time()

                distraction_duration = (
                    time.time()
                    -
                    data[
                        "distraction_start"
                    ]
                )

                if (
                    distraction_duration
                    >=
                    DISTRACTION_DELAY
                ):

                    status = (
                        "DISTRACTED"
                    )

                    # Count only once
                    # for this distraction period.
                    if not data[
                        "distraction_counted"
                    ]:

                        total_distraction_events += 1

                        data[
                            "distraction_counted"
                        ] = True

                        if (
                            name
                            !=
                            "Unknown"
                        ):

                            initialize_student(
                                name
                            )

                            student_stats[
                                name
                            ][
                                "distraction_events"
                            ] += 1

                else:

                    status = (
                        "CHECKING..."
                    )

            else:

                status = (
                    "ATTENTIVE"
                )

                data[
                    "distraction_start"
                ] = None

                data[
                    "distraction_counted"
                ] = False

            # =================================================
            # STUDENT STATISTICS
            # =================================================

            if (
                name
                !=
                "Unknown"
            ):

                update_student_stats(
                    name,
                    status == "ATTENTIVE",
                    distraction_duration
                )

            # =================================================
            # FACE BOX
            # =================================================

            landmarks = (
                face["landmarks"]
            )

            xs = [
                landmark.x
                for landmark
                in landmarks.landmark
            ]

            ys = [
                landmark.y
                for landmark
                in landmarks.landmark
            ]

            x1 = int(
                min(xs)
                *
                frame_width
            )

            y1 = int(
                min(ys)
                *
                frame_height
            )

            x2 = int(
                max(xs)
                *
                frame_width
            )

            y2 = int(
                max(ys)
                *
                frame_height
            )

            # Keep coordinates inside frame
            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                frame_width - 1,
                x2
            )

            y2 = min(
                frame_height - 1,
                y2
            )

            # =================================================
            # DISPLAY NAME
            # =================================================

            if name == "Unknown":

                display_name = (
                    "Unknown Student"
                )

            else:

                display_name = name

            # =================================================
            # DISPLAY LABEL
            # =================================================

            label = (
                f"{display_name} | {status}"
            )

            if status == "DISTRACTED":

                label += (
                    f" | "
                    f"{distraction_duration:.1f}s"
                )

            elif status == "CHECKING...":

                label += (
                    f" | "
                    f"{distraction_duration:.1f}s"
                )

            # =================================================
            # DRAW FACE BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            # =================================================
            # LABEL
            # =================================================

            text_y = max(
                25,
                y1 - 10
            )

            cv2.putText(
                frame,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            # =================================================
            # RECOGNITION INFORMATION
            # =================================================

            if name != "Unknown":

                cv2.putText(
                    frame,
                    f"Recognized | ML: {confidence:.0f}%",
                    (
                        x1,
                        min(
                            frame_height - 10,
                            y2 + 22
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1
                )

        # ====================================================
        # CURRENT STUDENT COUNT
        # ====================================================

        current_student_count = len(
            current_faces
        )

        if (
            current_student_count
            >
            maximum_students
        ):

            maximum_students = (
                current_student_count
            )

        # ====================================================
        # CLASS ATTENTION
        # ====================================================

        attentive_count = 0

        if current_student_count > 0:

            # Match current faces with their
            # tracked states.
            used_ids = set()

            for face in current_faces:

                best_id = None

                best_distance = float(
                    "inf"
                )

                for (
                    face_id,
                    data
                ) in face_history.items():

                    if face_id in used_ids:

                        continue

                    distance = (
                        calculate_distance(
                            face["center"],
                            data["center"]
                        )
                    )

                    if (
                        distance
                        <
                        best_distance
                    ):

                        best_distance = (
                            distance
                        )

                        best_id = (
                            face_id
                        )

                if best_id is None:

                    continue

                used_ids.add(
                    best_id
                )

                data = (
                    face_history[best_id]
                )

                if (
                    data[
                        "distraction_start"
                    ]
                    is None
                ):

                    attentive_count += 1

                else:

                    duration = (
                        time.time()
                        -
                        data[
                            "distraction_start"
                        ]
                    )

                    if (
                        duration
                        <
                        DISTRACTION_DELAY
                    ):

                        attentive_count += 1

            class_attention = (
                attentive_count
                /
                current_student_count
            ) * 100

        else:

            class_attention = 0.0

        attention_history.append(
            class_attention
        )

        # ====================================================
        # SESSION TIME
        # ====================================================

        session_time = (
            time.time()
            -
            session_start
        )

        # ====================================================
        # TOP INFORMATION BAR
        # ====================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (frame_width, 115),
            (0, 0, 0),
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
            f"Students: {current_student_count}",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Class Attention: {class_attention:.1f}%",
            (220, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Session: {session_time:.0f}s",
            (520, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "E: Enroll Student | Q: Quit & Save Report",
            (20, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "ClassFocus - Classroom Attention",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # ====================================================
        # ENROLL STUDENT
        # ====================================================

        if key == ord("e"):

            enroll_student(cap)

        # ====================================================
        # QUIT
        # ====================================================

        elif key == ord("q"):

            break

    # ========================================================
    # END SESSION
    # ========================================================

    session_duration = (
        time.time()
        -
        session_start
    )

    cap.release()

    cv2.destroyAllWindows()

    # ========================================================
    # SAVE REPORT
    # ========================================================

    save_report(
        session_duration,
        maximum_students,
        attention_history,
        total_distraction_events
    )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "              CLASSFOCUS SUMMARY"
    )

    print(
        "=============================================="
    )

    print(
        f"Maximum students detected: "
        f"{maximum_students}"
    )

    if len(
        attention_history
    ) > 0:

        print(
            f"Average class attention: "
            f"{np.mean(attention_history):.2f}%"
        )

        print(
            f"Peak class attention: "
            f"{np.max(attention_history):.2f}%"
        )

        print(
            f"Lowest class attention: "
            f"{np.min(attention_history):.2f}%"
        )

    print(
        f"Total distraction events: "
        f"{total_distraction_events}"
    )

    print(
        f"Session duration: "
        f"{session_duration:.1f} seconds"
    )

    print()

    # ========================================================
    # INDIVIDUAL RESULTS
    # ========================================================

    print(
        "Individual student results:"
    )

    if len(student_stats) == 0:

        print(
            "  No enrolled students were tracked."
        )

    for name, stats in (
        student_stats.items()
    ):

        if stats["frames"] > 0:

            percentage = (
                stats[
                    "attentive_frames"
                ]
                /
                stats["frames"]
            ) * 100

        else:

            percentage = 0

        print(
            f"  {name}: "
            f"{percentage:.1f}% attention | "
            f"{stats['distraction_events']} "
            f"distraction events"
        )

    print()

    print(
        "=============================================="
    )

    print(
        "Program finished."
    )

    print(
        "=============================================="
    )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()