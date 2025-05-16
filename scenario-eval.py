# Standard library imports
import os
import time
import pickle
from datetime import datetime

# Computer vision and numerical libraries
import cv2
import numpy as np
import torch
from PIL import Image

# Face recognition models
from facenet_pytorch import MTCNN, InceptionResnetV1

# Machine learning utilities
from sklearn.metrics.pairwise import cosine_distances

# OS-specific utility for sound feedback (Windows only)
import winsound

# GUI libraries for user input dialogs
import tkinter as tk
from tkinter import simpledialog


# =====================================
# Setup: Models and Storage
# =====================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=20, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
db_path = "registered_users.pkl"


# =====================================
# Helper: Load/Save User Database
# =====================================
def load_user_db(path=db_path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}

def save_user_db(user_db, path=db_path):
    with open(path, 'wb') as f:
        pickle.dump(user_db, f)

# =====================================
# Helper: Get Face Embedding from Frame
# =====================================
def get_face_embedding(frame):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    aligned = mtcnn(img_pil)

    if aligned is None:
        return None

    aligned = aligned.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = facenet(aligned).cpu().numpy()[0]
    return emb

# =====================================
# Helper: Register New User
# =====================================
def register_user(name, embeddings, user_db, threshold=0.3):
    if len(embeddings) == 0:
        print("No valid face captured. Registration failed.")
        return False

    mean_emb = np.mean(embeddings, axis=0)

    if len(user_db) > 0:
        db_embs = np.array(list(user_db.values()))
        distances = cosine_distances([mean_emb], db_embs)[0]
        min_dist = distances.min()
        if min_dist < threshold:
            print(f"Face too similar to existing user (distance {min_dist:.3f}). Registration aborted.")
            return False

    user_db[name] = mean_emb
    save_user_db(user_db)
    print(f"Registered: {name}")
    return True

# =====================================
# Helper: Recognize Face
# =====================================
def recognize_face(embedding, user_db, threshold=0.3):
    if embedding is None:
        return "No face detected", None

    if len(user_db) == 0:
        return "No users registered", None

    names = list(user_db.keys())
    db_embs = np.array(list(user_db.values()))

    distances = cosine_distances([embedding], db_embs)[0]
    min_dist = distances.min()
    min_idx = distances.argmin()

    if min_dist < threshold:
        return names[min_idx], min_dist
    else:
        return "Unknown", min_dist

# --- Sound functions ---
def play_access_granted():
    # Simple Windows beep: freq=1000Hz, duration=300ms
    winsound.Beep(1000, 300)

def play_access_denied():
    # Lower freq beep for denial
    winsound.Beep(400, 700)

# --- Logging function ---
def log_access(name, status, dist):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("access_log.csv", "a") as f:
        f.write(f"{now},{name},{status},{dist if dist is not None else 'N/A'}\n")

# --- GUI functions ---
def ask_username():
    root = tk.Tk()
    root.withdraw()  # Hide main window
    user_name = simpledialog.askstring(title="User Registration",
                                       prompt="Enter new user's name:")
    root.destroy()
    return user_name

# =====================================
# Main App Loop
# =====================================
def main():
    user_db = load_user_db()
    cap = cv2.VideoCapture(0)
    mode = "idle"
    registration_name = None
    registration_embs = []
    last_capture_time = 0
    capture_interval = 1.5  # seconds
    max_captures = 10
    instructions = [
        "Press 'a' to request access",
        "Press 'r' to register a new user",
        "Press 'q' to quit"
    ]
    status_message = ""
    status_message_time = 0
    status_message_duration = 3  # seconds to show message

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        current_time = time.time()

        # Show status message if active
        if status_message and (current_time - status_message_time) < status_message_duration:
            cv2.putText(frame, status_message, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            status_message = ""

        if mode == "registration":
            cv2.putText(frame, f"Registering: {registration_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.putText(frame, f"Please slowly turn your head left and right.", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # Capture embedding every 'capture_interval' seconds
            if current_time - last_capture_time > capture_interval and len(registration_embs) < max_captures:
                emb = get_face_embedding(frame)
                if emb is not None:
                    registration_embs.append(emb)
                    print(f"Captured embedding {len(registration_embs)}/{max_captures}")
                last_capture_time = current_time

            if len(registration_embs) >= max_captures:
                if not register_user(registration_name, registration_embs, user_db):
                    status_message = "Face already registered!"
                    status_message_time = time.time()
                else:
                    status_message = "Registration successful!"
                    status_message_time = time.time()

                registration_name = None
                registration_embs = []
                mode = "idle"

        elif mode == "access_check":
            cv2.putText(frame, f"Checking access...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            emb = get_face_embedding(frame)
            name, dist = recognize_face(emb, user_db)

            if name == "Unknown" or name == "No face detected":
                color = (0, 0, 255)
                access_status = "Access Denied"
                play_access_denied()
            else:
                color = (0, 255, 0)
                access_status = "Access Granted"
                play_access_granted()
                # Optional: trigger door unlock here

            log_access(name, access_status, dist)

            label = f"{name} - {access_status}"
            cv2.putText(frame, label, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("Access Control", frame)
            cv2.waitKey(3000)  # Show for 3 seconds
            mode = "idle"
            continue

        else:  # idle mode
            y0, dy = 30, 30  # Starting y position and line spacing
            for i, line in enumerate(instructions):
                y = y0 + i*dy
                cv2.putText(frame, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow("Access Control", frame)

        if key == ord('q'):
            break
        elif key == ord('r') and mode == "idle":
            registration_name = ask_username()
            if not registration_name:
                print("Registration cancelled.")
                status_message = "Registration cancelled."
                status_message_time = time.time()
                mode = "idle"
            elif registration_name in user_db:
                print(f"User '{registration_name}' is already registered.")
                status_message = f"User '{registration_name}' already registered!"
                status_message_time = time.time()
                mode = "idle"
            else:
                registration_embs = []
                mode = "registration"
                print(f"Starting registration for: {registration_name}")
        elif key == ord('a') and mode == "idle":
            mode = "access_check"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
