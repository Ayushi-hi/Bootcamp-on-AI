import cv2
import time
import csv
import math


# =========================================================
# CLASSFOCUS
# Simple Classroom Attention Detection
# =========================================================

STUDENT_REPORT = "classfocus_report.csv"
CLASS_REPORT = "classfocus_class_report.csv"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

MOVEMENT_THRESHOLD = 35
DISTRACTION_DELAY = 2.0


# =========================================================
# FACE DETECTOR
# =========================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# =========================================================
# STUDENT DATA
# =========================================================

students = {}
next_student_id = 1


# =========================================================
# CREATE STUDENT
# =========================================================

def create_student(x, y):

    global next_student_id

    student_id = next_student_id
    next_student_id += 1

    students[student_id] = {
        "x": x,
        "y": y,

        "first_seen": time.time(),
        "last_seen": time.time(),
        "last_update": time.time(),

        "total_time": 0.0,
        "attentive_time": 0.0,
        "distraction_time": 0.0,

        "distraction_start": None,
        "distraction_events": 0,

        "longest_distraction": 0.0
    }

    return student_id


# =========================================================
# FIND EXISTING STUDENT
# =========================================================

def find_student(x, y):

    best_id = None
    best_distance = 999999

    for student_id, student in students.items():

        distance = math.sqrt(
            (x - student["x"]) ** 2 +
            (y - student["y"]) ** 2
        )

        if distance < best_distance:
            best_distance = distance
            best_id = student_id

    if best_distance < 120:
        return best_id

    return None


# =========================================================
# ATTENTION LEVEL
# =========================================================

def get_attention_level(attention_percentage):

    if attention_percentage >= 80:
        return "HIGH"

    elif attention_percentage >= 50:
        return "MEDIUM"

    else:
        return "LOW"


# =========================================================
# CHECK ATTENTION
# =========================================================

def check_attention(student, x, y):

    old_x = student["x"]
    old_y = student["y"]

    movement = math.sqrt(
        (x - old_x) ** 2 +
        (y - old_y) ** 2
    )

    student["x"] = x
    student["y"] = y

    # Camera center
    center_x = CAMERA_WIDTH // 2

    # Distance from center
    distance_from_center = abs(
        x - center_x
    )

    moving_too_much = (
        movement > MOVEMENT_THRESHOLD
    )

    looking_away = (
        distance_from_center > 220
    )

    if moving_too_much or looking_away:
        return False

    return True


# =========================================================
# SAVE REPORTS
# =========================================================

def save_reports(session_duration, attention_history):

    print()
    print("=" * 60)
    print("                 CLASSFOCUS REPORT")
    print("=" * 60)

    # -----------------------------------------------------
    # CLASS STATISTICS
    # -----------------------------------------------------

    if attention_history:

        average_class_attention = (
            sum(attention_history) /
            len(attention_history)
        )

        peak_class_attention = max(
            attention_history
        )

        lowest_class_attention = min(
            attention_history
        )

    else:

        average_class_attention = 0
        peak_class_attention = 0
        lowest_class_attention = 0

    total_distraction_events = sum(
        student["distraction_events"]
        for student in students.values()
    )

    maximum_students = len(students)

    # -----------------------------------------------------
    # STUDENT REPORT
    # -----------------------------------------------------

    with open(
        STUDENT_REPORT,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Student",
            "Total Time (sec)",
            "Attentive Time (sec)",
            "Distraction Time (sec)",
            "Attention (%)",
            "Attention Level",
            "Distraction Events",
            "Longest Distraction (sec)"
        ])

        for student_id, student in students.items():

            total_time = student["total_time"]

            attentive_time = student["attentive_time"]

            distraction_time = student["distraction_time"]

            if total_time > 0:

                attention_percentage = (
                    attentive_time /
                    total_time
                ) * 100

            else:

                attention_percentage = 0

            level = get_attention_level(
                attention_percentage
            )

            writer.writerow([
                f"Student {student_id}",
                round(total_time, 2),
                round(attentive_time, 2),
                round(distraction_time, 2),
                round(attention_percentage, 2),
                level,
                student["distraction_events"],
                round(
                    student["longest_distraction"],
                    2
                )
            ])

    # -----------------------------------------------------
    # CLASS REPORT
    # -----------------------------------------------------

    with open(
        CLASS_REPORT,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Session Duration (sec)",
            "Students Detected",
            "Average Class Attention (%)",
            "Peak Class Attention (%)",
            "Lowest Class Attention (%)",
            "Total Distraction Events"
        ])

        writer.writerow([
            round(session_duration, 2),
            maximum_students,
            round(
                average_class_attention,
                2
            ),
            round(
                peak_class_attention,
                2
            ),
            round(
                lowest_class_attention,
                2
            ),
            total_distraction_events
        ])

    # -----------------------------------------------------
    # PRINT STUDENT REPORT
    # -----------------------------------------------------

    print()
    print("STUDENT REPORT")
    print("-" * 60)

    for student_id, student in students.items():

        total_time = student["total_time"]

        if total_time > 0:

            attention_percentage = (
                student["attentive_time"] /
                total_time
            ) * 100

        else:

            attention_percentage = 0

        level = get_attention_level(
            attention_percentage
        )

        print(
            f"Student {student_id}: "
            f"{attention_percentage:.1f}% | "
            f"{level} | "
            f"Distractions: "
            f"{student['distraction_events']}"
        )

    # -----------------------------------------------------
    # PRINT CLASS REPORT
    # -----------------------------------------------------

    print()
    print("CLASS REPORT")
    print("-" * 60)

    print(
        f"Session Duration: "
        f"{session_duration:.1f} seconds"
    )

    print(
        f"Students Detected: "
        f"{maximum_students}"
    )

    print(
        f"Average Class Attention: "
        f"{average_class_attention:.1f}%"
    )

    print(
        f"Peak Class Attention: "
        f"{peak_class_attention:.1f}%"
    )

    print(
        f"Lowest Class Attention: "
        f"{lowest_class_attention:.1f}%"
    )

    print(
        f"Total Distraction Events: "
        f"{total_distraction_events}"
    )

    print()
    print("Reports created:")
    print(f"- {STUDENT_REPORT}")
    print(f"- {CLASS_REPORT}")

    print("=" * 60)


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 60)
    print("                    CLASSFOCUS")
    print("          Classroom Attention Detection")
    print("=" * 60)

    print()
    print("Starting camera...")
    print()
    print("Q = End session and generate reports")
    print()

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    if not cap.isOpened():

        print("ERROR: Camera could not be opened.")
        return

    session_start = time.time()

    attention_history = []

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Could not read camera.")
            break

        current_time = time.time()

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # -------------------------------------------------
        # DETECT FACES
        # -------------------------------------------------

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        current_attention_values = []

        # -------------------------------------------------
        # PROCESS FACES
        # -------------------------------------------------

        for (x, y, w, h) in faces:

            center_x = x + w // 2
            center_y = y + h // 2

            student_id = find_student(
                center_x,
                center_y
            )

            if student_id is None:

                student_id = create_student(
                    center_x,
                    center_y
                )

            student = students[student_id]

            # Time since last frame
            delta = (
                current_time -
                student["last_update"]
            )

            if delta > 1:
                delta = 0

            student["total_time"] += delta
            student["last_update"] = current_time
            student["last_seen"] = current_time

            # -------------------------------------------------
            # ATTENTION
            # -------------------------------------------------

            attentive = check_attention(
                student,
                center_x,
                center_y
            )

            if attentive:

                student["attentive_time"] += delta

                # Stop distraction timer
                student["distraction_start"] = None

            else:

                if student["distraction_start"] is None:

                    student["distraction_start"] = (
                        current_time
                    )

                distraction_duration = (
                    current_time -
                    student["distraction_start"]
                )

                # Start counting after 2 seconds
                if distraction_duration >= DISTRACTION_DELAY:

                    student["distraction_time"] += delta

                    # Count event once
                    if (
                        distraction_duration - delta
                        < DISTRACTION_DELAY
                    ):

                        student["distraction_events"] += 1

                if distraction_duration > student[
                    "longest_distraction"
                ]:

                    student[
                        "longest_distraction"
                    ] = distraction_duration

            # -------------------------------------------------
            # ATTENTION %
            # -------------------------------------------------

            if student["total_time"] > 0:

                attention_percentage = (
                    student["attentive_time"] /
                    student["total_time"]
                ) * 100

            else:

                attention_percentage = 0

            attention_percentage = max(
                0,
                min(
                    100,
                    attention_percentage
                )
            )

            current_attention_values.append(
                attention_percentage
            )

            # -------------------------------------------------
            # ATTENTION LEVEL
            # -------------------------------------------------

            level = get_attention_level(
                attention_percentage
            )

            # -------------------------------------------------
            # STATUS
            # -------------------------------------------------

            if attentive:

                status = "ATTENTIVE"

            else:

                distraction_duration = (
                    current_time -
                    student["distraction_start"]
                )

                if (
                    distraction_duration
                    >= DISTRACTION_DELAY
                ):

                    status = "DISTRACTED"

                else:

                    status = "CHECKING..."

            # -------------------------------------------------
            # COLORS
            # -------------------------------------------------

            if level == "HIGH":

                color = (0, 255, 0)

            elif level == "MEDIUM":

                color = (0, 255, 255)

            else:

                color = (0, 0, 255)

            # -------------------------------------------------
            # FACE BOX
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            # Student
            cv2.putText(
                frame,
                f"Student {student_id}",
                (x, y - 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

            # Attention
            cv2.putText(
                frame,
                f"Attention: {attention_percentage:.0f}%",
                (x, y - 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            # Level
            cv2.putText(
                frame,
                f"Level: {level}",
                (x, y - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            # Status
            cv2.putText(
                frame,
                status,
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            # -------------------------------------------------
            # DISTRACTION TIMER
            # -------------------------------------------------

            if student["distraction_start"] is not None:

                distraction_duration = (
                    current_time -
                    student["distraction_start"]
                )

                if distraction_duration >= 0.5:

                    timer_text = (
                        f"Distraction: "
                        f"{distraction_duration:.1f}s"
                    )

                    cv2.putText(
                        frame,
                        timer_text,
                        (x, y + h + 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )

        # -------------------------------------------------
        # CLASS ATTENTION
        # -------------------------------------------------

        if current_attention_values:

            class_attention = (
                sum(current_attention_values) /
                len(current_attention_values)
            )

            attention_history.append(
                class_attention
            )

        else:

            class_attention = 0

        # -------------------------------------------------
        # CLASS LEVEL
        # -------------------------------------------------

        class_level = get_attention_level(
            class_attention
        )

        # -------------------------------------------------
        # SESSION TIMER
        # -------------------------------------------------

        session_time = (
            current_time -
            session_start
        )

        minutes = int(session_time // 60)

        seconds = int(session_time % 60)

        # -------------------------------------------------
        # DASHBOARD
        # -------------------------------------------------

        cv2.rectangle(
            frame,
            (0, 0),
            (640, 65),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            "CLASSFOCUS",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Students: {len(faces)}",
            (150, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Class: {class_attention:.0f}%",
            (275, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            class_level,
            (390, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"{minutes:02d}:{seconds:02d}",
            (490, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q=Report",
            (500, 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1
        )

        # -------------------------------------------------
        # SHOW
        # -------------------------------------------------

        cv2.imshow(
            "ClassFocus - Classroom Attention",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

    # =====================================================
    # END SESSION
    # =====================================================

    session_duration = (
        time.time() -
        session_start
    )

    cap.release()

    cv2.destroyAllWindows()

    # Automatically generate reports
    save_reports(
        session_duration,
        attention_history
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    main()