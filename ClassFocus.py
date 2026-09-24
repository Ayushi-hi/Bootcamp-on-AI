import cv2
import time
import csv
import math


# =========================================================
# CONFIGURATION
# =========================================================

STUDENT_REPORT = "classfocus_report.csv"
CLASS_REPORT = "classfocus_class_report.csv"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Movement thresholds
ATTENTIVE_MOVEMENT = 8
LESS_ATTENTIVE_MOVEMENT = 35

# How long movement must continue before distraction
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


# =========================================================
# CREATE NEW STUDENT
# =========================================================

def create_student(student_id, x, y):

    current_time = time.time()

    return {

        "id": student_id,

        "x": x,
        "y": y,

        "first_seen": current_time,
        "last_seen": current_time,
        "last_update": current_time,

        # Time statistics
        "total_time": 0,

        "attentive_time": 0,
        "less_attentive_time": 0,
        "distracted_time": 0,

        # Distraction information
        "distraction_start": None,
        "distraction_events": 0,
        "longest_distraction": 0,

        # Current status
        "last_status": "ATTENTIVE",
        "attention_percentage": 100
    }


# =========================================================
# FIND EXISTING STUDENT
# =========================================================

def find_student(x, y):

    closest_student = None
    closest_distance = float("inf")

    for student_id, student in students.items():

        distance = math.sqrt(
            (x - student["x"]) ** 2 +
            (y - student["y"]) ** 2
        )

        if distance < 120 and distance < closest_distance:

            closest_distance = distance
            closest_student = student_id

    return closest_student


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

    movement_x = x - old_x
    movement_y = y - old_y

    movement = math.sqrt(
        movement_x ** 2 +
        movement_y ** 2
    )

    # Update position
    student["x"] = x
    student["y"] = y


    # =====================================================
    # DETERMINE MOVEMENT DIRECTION
    # =====================================================

    if abs(movement_x) > abs(movement_y):

        if movement_x > 0:

            direction = "RIGHT"

        else:

            direction = "LEFT"

    else:

        if movement_y > 0:

            direction = "DOWN"

        else:

            direction = "UP"


    # =====================================================
    # ATTENTIVE
    # =====================================================

    if movement <= ATTENTIVE_MOVEMENT:

        status = "ATTENTIVE"

        percentage = 95

        color = (0, 255, 0)


    # =====================================================
    # LESS ATTENTIVE
    # =====================================================

    elif movement <= LESS_ATTENTIVE_MOVEMENT:

        status = "LESS ATTENTIVE"

        percentage = int(
            90 -
            (
                (movement - ATTENTIVE_MOVEMENT)
                /
                (
                    LESS_ATTENTIVE_MOVEMENT -
                    ATTENTIVE_MOVEMENT
                )
            )
            * 40
        )

        color = (0, 255, 255)


    # =====================================================
    # DISTRACTED
    # =====================================================

    else:

        status = "DISTRACTED"

        percentage = int(
            max(
                0,
                45 -
                (
                    movement -
                    LESS_ATTENTIVE_MOVEMENT
                )
            )
        )

        color = (0, 0, 255)


    return (
        status,
        percentage,
        color,
        movement,
        direction
    )


# =========================================================
# UPDATE STUDENT STATISTICS
# =========================================================

def update_student_statistics(
    student,
    status,
    current_time
):

    elapsed = (
        current_time -
        student["last_update"]
    )

    if elapsed < 0:

        elapsed = 0


    # =====================================================
    # TOTAL TIME
    # =====================================================

    student["total_time"] += elapsed


    # =====================================================
    # ATTENTIVE TIME
    # =====================================================

    if status == "ATTENTIVE":

        student["attentive_time"] += elapsed


    # =====================================================
    # LESS ATTENTIVE TIME
    # =====================================================

    elif status == "LESS ATTENTIVE":

        student["less_attentive_time"] += elapsed


    # =====================================================
    # DISTRACTED TIME
    # =====================================================

    elif status == "DISTRACTED":

        student["distracted_time"] += elapsed


    # =====================================================
    # DISTRACTION TRACKING
    # =====================================================

    if status in [
        "LESS ATTENTIVE",
        "DISTRACTED"
    ]:

        if student["distraction_start"] is None:

            student["distraction_start"] = current_time

            student["distraction_events"] += 1

    else:

        if student["distraction_start"] is not None:

            distraction_duration = (
                current_time -
                student["distraction_start"]
            )

            if (
                distraction_duration >
                student["longest_distraction"]
            ):

                student["longest_distraction"] = (
                    distraction_duration
                )

            student["distraction_start"] = None


    student["last_update"] = current_time
    student["last_seen"] = current_time
    student["last_status"] = status


# =========================================================
# CLOSE OPEN DISTRACTIONS BEFORE SAVING
# =========================================================

def close_open_distractions(current_time):

    for student in students.values():

        if student["distraction_start"] is not None:

            distraction_duration = (
                current_time -
                student["distraction_start"]
            )

            if (
                distraction_duration >
                student["longest_distraction"]
            ):

                student["longest_distraction"] = (
                    distraction_duration
                )

            student["distraction_start"] = None


# =========================================================
# SAVE STUDENT REPORT
# =========================================================

def save_student_report():

    with open(
        STUDENT_REPORT,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([

            "Student",

            "Total Time (seconds)",

            "Attentive Time (seconds)",

            "Less Attentive Time (seconds)",

            "Distracted Time (seconds)",

            "Attention %",

            "Attention Level",

            "Distraction Events",

            "Longest Distraction (seconds)"
        ])


        for student_id, student in students.items():

            total_time = student["total_time"]


            # =================================================
            # OVERALL ATTENTION %
            # =================================================

            if total_time > 0:

                attention_percentage = (
                    student["attentive_time"]
                    /
                    total_time
                ) * 100

            else:

                attention_percentage = 0


            attention_level = get_attention_level(
                attention_percentage
            )


            writer.writerow([

                f"Student {student_id}",

                round(
                    total_time,
                    2
                ),

                round(
                    student["attentive_time"],
                    2
                ),

                round(
                    student["less_attentive_time"],
                    2
                ),

                round(
                    student["distracted_time"],
                    2
                ),

                round(
                    attention_percentage,
                    2
                ),

                attention_level,

                student["distraction_events"],

                round(
                    student["longest_distraction"],
                    2
                )
            ])


# =========================================================
# SAVE CLASS REPORT
# =========================================================

def save_class_report(session_duration):

    if len(students) == 0:

        average_attention = 0
        peak_attention = 0
        lowest_attention = 0
        total_distraction_events = 0

    else:

        attention_values = []

        total_distraction_events = 0


        for student in students.values():

            total_time = student["total_time"]

            if total_time > 0:

                attention = (
                    student["attentive_time"]
                    /
                    total_time
                ) * 100

            else:

                attention = 0


            attention_values.append(
                attention
            )

            total_distraction_events += (
                student["distraction_events"]
            )


        average_attention = (
            sum(attention_values)
            /
            len(attention_values)
        )

        peak_attention = max(
            attention_values
        )

        lowest_attention = min(
            attention_values
        )


    with open(
        CLASS_REPORT,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([

            "Session Duration (seconds)",

            "Students Detected",

            "Average Class Attention %",

            "Peak Attention %",

            "Lowest Attention %",

            "Total Distraction Events"
        ])


        writer.writerow([

            round(
                session_duration,
                2
            ),

            len(students),

            round(
                average_attention,
                2
            ),

            round(
                peak_attention,
                2
            ),

            round(
                lowest_attention,
                2
            ),

            total_distraction_events
        ])


# =========================================================
# PRINT REPORT
# =========================================================

def print_report(session_duration):

    print("\n")
    print("=" * 70)
    print("              CLASSFOCUS SESSION REPORT")
    print("=" * 70)

    print(
        f"Session Duration: "
        f"{session_duration:.2f} seconds"
    )

    print(
        f"Students Detected: "
        f"{len(students)}"
    )

    print("-" * 70)


    for student_id, student in students.items():

        total_time = student["total_time"]


        if total_time > 0:

            attention_percentage = (
                student["attentive_time"]
                /
                total_time
            ) * 100

        else:

            attention_percentage = 0


        print(
            f"\nStudent {student_id}"
        )

        print(
            f"  Total Time: "
            f"{total_time:.2f} seconds"
        )

        print(
            f"  ATTENTIVE: "
            f"{student['attentive_time']:.2f} seconds"
        )

        print(
            f"  LESS ATTENTIVE: "
            f"{student['less_attentive_time']:.2f} seconds"
        )

        print(
            f"  DISTRACTED: "
            f"{student['distracted_time']:.2f} seconds"
        )

        print(
            f"  Attention: "
            f"{attention_percentage:.2f}%"
        )

        print(
            f"  Attention Level: "
            f"{get_attention_level(attention_percentage)}"
        )

        print(
            f"  Distraction Events: "
            f"{student['distraction_events']}"
        )

        print(
            f"  Longest Distraction: "
            f"{student['longest_distraction']:.2f} seconds"
        )


    print("\n")
    print("=" * 70)

    print("Reports saved:")

    print(
        f"1. {STUDENT_REPORT}"
    )

    print(
        f"2. {CLASS_REPORT}"
    )

    print("=" * 70)


# =========================================================
# MAIN
# =========================================================

def main():

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

        print(
            "Could not open camera."
        )

        return


    session_start = time.time()

    attention_history = []


    print(
        "ClassFocus started."
    )

    print(
        "Press Q to quit."
    )

    print(
        "Press Ctrl+C to stop and save the reports."
    )


    try:

        while True:

            ret, frame = cap.read()


            if not ret:

                print(
                    "Could not read camera."
                )

                time.sleep(0.1)

                continue


            # =================================================
            # MIRROR CAMERA
            # =================================================

            frame = cv2.flip(
                frame,
                1
            )


            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )


            # =================================================
            # DETECT FACES
            # =================================================

            faces = face_detector.detectMultiScale(

                gray,

                scaleFactor=1.1,

                minNeighbors=5,

                minSize=(60, 60)
            )


            current_time = time.time()

            current_attention_values = []


            # =================================================
            # PROCESS EACH STUDENT
            # =================================================

            for (
                x,
                y,
                w,
                h
            ) in faces:


                center_x = x + w // 2

                center_y = y + h // 2


                student_id = find_student(

                    center_x,

                    center_y
                )


                # =============================================
                # CREATE NEW STUDENT
                # =============================================

                if student_id is None:

                    student_id = (
                        len(students) + 1
                    )

                    students[student_id] = (
                        create_student(

                            student_id,

                            center_x,

                            center_y
                        )
                    )


                student = students[student_id]


                # =============================================
                # CHECK ATTENTION
                # =============================================

                (
                    status,
                    percentage,
                    color,
                    movement,
                    direction
                ) = check_attention(

                    student,

                    center_x,

                    center_y
                )


                student[
                    "attention_percentage"
                ] = percentage


                # =============================================
                # UPDATE STATISTICS
                # =============================================

                update_student_statistics(

                    student,

                    status,

                    current_time
                )


                current_attention_values.append(
                    percentage
                )


                attention_history.append(
                    percentage
                )


                # =============================================
                # DRAW FACE BOX
                # =============================================

                cv2.rectangle(

                    frame,

                    (x, y),

                    (x + w, y + h),

                    color,

                    2
                )


                # =============================================
                # STUDENT LABEL
                # =============================================

                cv2.putText(

                    frame,

                    f"Student {student_id}",

                    (
                        x,
                        y - 45
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    color,

                    2
                )


                # =============================================
                # STATUS LABEL
                # =============================================

                cv2.putText(

                    frame,

                    f"{status} - {percentage}%",

                    (
                        x,
                        y - 20
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    color,

                    2
                )


                # =============================================
                # HEAD DIRECTION
                # =============================================

                if status == "LESS ATTENTIVE":

                    cv2.putText(

                        frame,

                        f"Head: {direction}",

                        (
                            x,
                            y + h + 20
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.5,

                        color,

                        2
                    )


                # =============================================
                # MOVEMENT VALUE
                # =============================================

                cv2.putText(

                    frame,

                    f"Movement: {movement:.1f}",

                    (
                        x,
                        y + h + 42
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.45,

                    color,

                    1
                )


            # =================================================
            # CLASS ATTENTION
            # =================================================

            if len(current_attention_values) > 0:

                class_attention = (
                    sum(current_attention_values)
                    /
                    len(current_attention_values)
                )

            else:

                class_attention = 0


            # =================================================
            # DASHBOARD
            # =================================================

            cv2.rectangle(

                frame,

                (10, 10),

                (260, 105),

                (30, 30, 30),

                -1
            )


            cv2.putText(

                frame,

                f"Students: {len(students)}",

                (20, 35),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 255, 255),

                2
            )


            cv2.putText(

                frame,

                f"Class Attention: {class_attention:.1f}%",

                (20, 60),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (255, 255, 255),

                2
            )


            cv2.putText(

                frame,

                "Press Q to Quit",

                (20, 88),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.45,

                (200, 200, 200),

                1
            )


            # =================================================
            # LEGEND
            # =================================================

            cv2.putText(

                frame,

                "GREEN = ATTENTIVE",

                (
                    10,
                    CAMERA_HEIGHT - 70
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (0, 255, 0),

                2
            )


            cv2.putText(

                frame,

                "YELLOW = LESS ATTENTIVE",

                (
                    10,
                    CAMERA_HEIGHT - 45
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (0, 255, 255),

                2
            )


            cv2.putText(

                frame,

                "RED = DISTRACTED",

                (
                    10,
                    CAMERA_HEIGHT - 20
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (0, 0, 255),

                2
            )


            # =================================================
            # SHOW WINDOW
            # =================================================

            cv2.imshow(

                "ClassFocus - Classroom Attention",

                frame
            )


            # =================================================
            # QUIT
            # =================================================

            key = cv2.waitKey(1) & 0xFF


            if key == ord("q"):

                break


    except KeyboardInterrupt:

        print("\n")
        print("ClassFocus stopped with Ctrl+C.")
        print("Saving reports...")


    finally:

        # =====================================================
        # END SESSION
        # =====================================================

        session_end = time.time()

        session_duration = (
            session_end -
            session_start
        )


        # =====================================================
        # CLOSE OPEN DISTRACTION PERIODS
        # =====================================================

        close_open_distractions(
            session_end
        )


        # =====================================================
        # ADD FINAL TIME FOR CURRENT STUDENTS
        # =====================================================

        for student in students.values():

            if student["last_seen"] <= session_end:

                final_elapsed = (
                    session_end -
                    student["last_update"]
                )

                if final_elapsed > 0:

                    if student["last_status"] == "ATTENTIVE":

                        student["attentive_time"] += (
                            final_elapsed
                        )

                    elif (
                        student["last_status"]
                        ==
                        "LESS ATTENTIVE"
                    ):

                        student["less_attentive_time"] += (
                            final_elapsed
                        )

                    elif (
                        student["last_status"]
                        ==
                        "DISTRACTED"
                    ):

                        student["distracted_time"] += (
                            final_elapsed
                        )

                    student["total_time"] += (
                        final_elapsed
                    )


        # =====================================================
        # RELEASE CAMERA
        # =====================================================

        cap.release()

        cv2.destroyAllWindows()


        # =====================================================
        # SAVE REPORTS
        # =====================================================

        save_student_report()

        save_class_report(
            session_duration
        )


        # =====================================================
        # PRINT REPORT
        # =====================================================

        print_report(
            session_duration
        )


# =========================================================
# START PROGRAM
# =========================================================

if __name__ == "__main__":

    main()