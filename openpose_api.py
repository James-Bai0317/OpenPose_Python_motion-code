import sys
import cv2
import os
import numpy as np
from sys import platform

BODY_25_PAIRS = {
    "right_elbow": (2, 3, 4),
    "left_elbow": (5, 6, 7),
    "right_knee": (9, 10, 11),
    "left_knee": (12, 13, 14),
    "right_H": (3, 1, 8),
    "left_H": (5, 1, 8),
    "right_F": (1, 8, 10),
    "left_F": (1, 8, 13)
}


def detect_fist(kp_matrix, angles):
    left_wrist_y = kp_matrix[7, 1]
    left_elbow_y = kp_matrix[6, 1]
    right_wrist_y = kp_matrix[4, 1]
    right_elbow_y = kp_matrix[3, 1]

    if angles["left_elbow"] > 140 or angles["right_elbow"] > 140:
        if left_wrist_y < left_elbow_y or right_wrist_y < right_elbow_y:
            return True
    return False


def calculate_angle(A, B, C):
    # A, B, C: (x, y) coordinates of joints
    # Returns angle at B in degrees

    BA = A - B
    BC = C - B

    # Avoid division by zero
    if np.linalg.norm(BA) == 0 or np.linalg.norm(BC) == 0:
        return np.nan

    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    # Clip to avoid floating point errors outside [-1, 1]
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cosine_angle))


def get_elbow_angle(keypoints):
    right_shoulder = keypoints[2, :2]  # (x, y)
    right_elbow = keypoints[3, :2]
    right_wrist = keypoints[4, :2]

    return calculate_angle(right_shoulder, right_elbow, right_wrist)


# --- Import OpenPose ---
OPENPOSE_ROOT = r"C:\openpose-1.7.0"

pyopenpose_path = os.path.join(OPENPOSE_ROOT, "build", "python", "openpose", "Release")

# 改成 insert(0, ...)，把 OpenPose 路徑放在最前面
if pyopenpose_path not in sys.path:
    sys.path.insert(0, pyopenpose_path)

# DLL 路徑
os.environ["PATH"] = os.path.join(OPENPOSE_ROOT, "build", "x64", "Release") + ";" + os.path.join(OPENPOSE_ROOT, "bin") + ";" + os.environ["PATH"]

try:
    import pyopenpose as op
    print("✅ pyopenpose 載入成功！")
except Exception as e:
    print("❌ 載入失敗：", e)

# --- OpenPose Parameters ---
params = dict()
params["model_folder"] = "C:/openpose-1.7.0/models/"

# --- Initialize OpenPose ---
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# --- Video Capture ---
# video_path = "E:/HomeWork/ChuanT/openpose-1.5.1/examples/media/hittest.mp4"
# cap = cv2.VideoCapture(video_path)

# --- Webcam Capture ---
cap = cv2.VideoCapture(0)  # 0 = built-in/default webcam, 1 or 2 if you have multiple
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Create datum & process frame
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    keypoints = datum.poseKeypoints  # shape: [people, 25, 3]

    if keypoints is not None:
        try:
            for person_id in range(keypoints.shape[0]):
                kp_matrix = keypoints[person_id]  # Shape: (25, 3)

                angles = {}
                for name, (a, b, c) in BODY_25_PAIRS.items():
                    angles[name] = calculate_angle(kp_matrix[a, :2], kp_matrix[b, :2], kp_matrix[c, :2])

                print(person_id)
                print(angles)

                # motion detection
                if detect_fist(kp_matrix, angles):
                    print("Successfully hitted!")


                # Save keypoints per frame per person
                # filename = f"frame_{frame_id:05d}_person_{person_id+1}.csv"
                # np.savetxt(filename, kp_matrix, delimiter=",", header="x,y,confidence", comments='')
        except:
            try:
                kp_matrix = keypoints[person_id]  # Shape: (25, 3)

                angles = {}
                for name, (a, b, c) in BODY_25_PAIRS.items():
                    angles[name] = calculate_angle(kp_matrix[a, :2], kp_matrix[b, :2], kp_matrix[c, :2])

                print(angles)
            except:
                print("nobody in the picture!")

    # Optional: Show output
    cv2.imshow("OpenPose Video", datum.cvOutputData)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
