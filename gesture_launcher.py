"""
gesture_launcher.py

Requirements:
    pip install opencv-python mediapipe

Optional:
    pip install pyttsx3   # for voice feedback

Usage:
    python gesture_launcher.py
Press 'q' to quit the webcam window.

Customize:
    - Edit APP_MAPPING to change what each finger-count does.
    - Edit STABLE_FRAMES and COOLDOWN_SECONDS to tune responsiveness.
"""

import cv2
import mediapipe as mp
import time
import webbrowser
import subprocess
import sys
import os
import shutil

# Optional voice feedback
try:
    import pyttsx3
    VOICE_AVAILABLE = True
    tts_engine = pyttsx3.init()
except Exception:
    VOICE_AVAILABLE = False

# ------------------------
# Configuration
# ------------------------

# How many consecutive frames the same finger count must appear before triggering
STABLE_FRAMES = 6

# After triggering, wait this many seconds before allowing the next trigger
COOLDOWN_SECONDS = 3.0

# Webcam index (0 is usually default)
WEBCAM_INDEX = 0

# Finger tip landmark indices (MediaPipe hand landmark indices)
FINGER_TIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

# Map finger_count -> action lambda / function / tuple.
# Provide simple actions: "open_url", "open_program", "say_text", "noop"
# Example mapping here:
APP_MAPPING = {
    0: ("noop", None),  # closed fist: do nothing
    1: ("open_url", "https://www.youtube.com/"),
    2: ("open_program", "chrome"),   # will try to find an executable
    3: ("open_program", "code"),     # VS Code if 'code' available
    4: ("open_program", "explorer" if sys.platform.startswith("win") else "nautilus"),
    5: ("open_program", None),       # five fingers: you can set to None or another app
}

# If a program key is ambiguous, we try a lookup list per platform.
PROGRAM_LOOKUP = {
    "chrome": {
        "win32": ["chrome", r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                  r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"],
        "darwin": ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"],
        "linux": ["google-chrome", "chrome", "chromium", "chromium-browser"]
    },
    "code": {
        "win32": ["code"],  # If VS Code 'code' in PATH
        "darwin": ["code", "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code"],
        "linux": ["code"]
    },
    "explorer": {
        "win32": ["explorer"],
        "darwin": ["open"],      # macOS 'open' can open Finder if used properly
        "linux": ["xdg-open", "nautilus", "nemo"]
    }
}

# ------------------------
# Utilities
# ------------------------

def speak(text: str):
    if not VOICE_AVAILABLE:
        return
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception:
        pass

def find_executable_for_key(key: str):
    """
    Given a PROGRAM_LOOKUP key, try to find a usable executable command.
    Returns the command string to execute (or None if not found).
    """
    plat = sys.platform
    lookup = PROGRAM_LOOKUP.get(key, {})
    # try exact platform key matching style
    for plat_key, candidates in lookup.items():
        if plat.startswith(plat_key):
            for c in candidates:
                # if absolute path exists
                if os.path.isabs(c) and os.path.exists(c):
                    return c
                # if on PATH
                path = shutil.which(c)
                if path:
                    return path
    # fallback: search all candidates across platforms
    for candidates in lookup.values():
        for c in candidates:
            if os.path.isabs(c) and os.path.exists(c):
                return c
            path = shutil.which(c)
            if path:
                return path
    return None

def launch_program_by_key(key: str):
    cmd = find_executable_for_key(key)
    if cmd:
        try:
            # On macOS, if cmd is '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            # just call it directly.
            subprocess.Popen([cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            # fallback: try opening via system open
            try:
                if sys.platform == "darwin":
                    subprocess.Popen(["open", "-a", key])
                    return True
                elif sys.platform.startswith("linux"):
                    subprocess.Popen([cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return True
            except Exception:
                print(f"[WARN] Could not launch {key}: {e}")
                return False
    else:
        print(f"[INFO] No executable found for '{key}'.")
        return False

def perform_mapped_action(mapping_tuple):
    """
    mapping_tuple is ("action_type", param)
    """
    action, param = mapping_tuple
    if action == "noop":
        return False
    if action == "open_url" and param:
        webbrowser.open(param)
        return True
    if action == "open_program":
        # param can be a direct executable path or a key to PROGRAM_LOOKUP
        if param is None:
            print("[INFO] open_program mapping has no program specified.")
            return False
        # If param is a path or exe in PATH
        if os.path.isabs(param) and os.path.exists(param):
            try:
                subprocess.Popen([param], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception as e:
                print(f"[WARN] Couldn't launch {param}: {e}")
                return False
        # Else treat param as key for PROGRAM_LOOKUP or executable name
        # Try direct 'which' first
        exe = shutil.which(param)
        if exe:
            try:
                subprocess.Popen([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception as e:
                print(f"[WARN] Couldn't launch {exe}: {e}")
        # Try PROGRAM_LOOKUP
        started = launch_program_by_key(param)
        return started
    if action == "say_text" and param:
        speak(param)
        return True

    print(f"[WARN] Unknown action '{action}' or missing parameter.")
    return False

# ------------------------
# Finger counting logic
# ------------------------

def count_fingers_from_landmarks(hand_landmarks, hand_label=None):
    """
    Input:
        hand_landmarks: mediapipe.landmark list (normalized coordinates)
        hand_label: 'Left' or 'Right' (string) or None
    Returns:
        int: number of extended fingers (0-5)
        list: list of 0/1 for [thumb, index, middle, ring, pinky]
    """
    # Convert landmarks to simpler list of (x,y)
    lm = [(lmpt.x, lmpt.y) for lmpt in hand_landmarks.landmark]

    fingers = []

    # Thumb: compare tip with IP (tip index 4, IP index 3)
    # Logic depends on handedness because thumb extends sideways
    try:
        tip_x = lm[FINGER_TIPS[0]][0]
        ip_x = lm[FINGER_TIPS[0] - 1][0]
        if hand_label is None:
            # fallback: assume right
            hand_label = "Right"
        # This logic works for most camera setups when the frame is flipped for mirror view
        if hand_label == "Right":
            # for right hand, thumb extended when tip_x < ip_x
            fingers.append(1 if tip_x < ip_x else 0)
        else:
            # for left hand, thumb extended when tip_x > ip_x
            fingers.append(1 if tip_x > ip_x else 0)
    except Exception:
        fingers.append(0)

    # Other fingers: tip.y < pip.y means finger is up (y increases downward)
    for i in range(1, 5):
        try:
            tip_y = lm[FINGER_TIPS[i]][1]
            pip_y = lm[FINGER_TIPS[i] - 2][1]  # pip joint is two indices before tip
            fingers.append(1 if tip_y < pip_y else 0)
        except Exception:
            fingers.append(0)

    return sum(fingers), fingers

# ------------------------
# Main loop
# ------------------------

def main():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check WEBCAM_INDEX or permissions.")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils

    last_trigger_time = 0.0
    stable_counter = 0
    last_count = None

    print("[INFO] Starting gesture listener. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed reading frame from webcam.")
            break

        # Mirror the frame so it feels natural (mirror view)
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        display_text = "No hand"

        if results.multi_hand_landmarks:
            # hand landmarks and handness (Left/Right) are parallel lists. Use the first hand.
            hand_landmarks = results.multi_hand_landmarks[0]

            # Get hand label
            hand_label = None
            if results.multi_handedness:
                try:
                    hand_label = results.multi_handedness[0].classification[0].label
                except Exception:
                    hand_label = None

            # Count fingers
            finger_count, fingers_list = count_fingers_from_landmarks(hand_landmarks, hand_label)
            display_text = f"Fingers: {finger_count}  pattern: {fingers_list}  hand: {hand_label}"

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Stability logic: require STABLE_FRAMES consecutive frames with same finger_count
            if finger_count == last_count:
                stable_counter += 1
            else:
                stable_counter = 1  # reset to 1 for the current frame
                last_count = finger_count

            # Trigger only if stable for required frames AND cooldown passed
            now = time.time()
            if stable_counter >= STABLE_FRAMES and (now - last_trigger_time) >= COOLDOWN_SECONDS:
                # Perform mapped action
                mapping = APP_MAPPING.get(finger_count, ("noop", None))
                acted = perform_mapped_action(mapping)
                if acted:
                    last_trigger_time = now
                    # Optional voice feedback
                    if mapping[0] == "open_url":
                        speak(f"Opening website.")
                    elif mapping[0] == "open_program":
                        speak(f"Opening program.")
                    elif mapping[0] == "say_text":
                        speak(mapping[1])
                    else:
                        # Generic
                        speak("Action performed.")

        else:
            # No hand detected; reset stability but don't reset last_count so quick return works
            stable_counter = 0
            last_count = None

        # HUD overlay
        cv2.rectangle(frame, (0,0), (frame.shape[1], 30), (0,0,0), -1)  # top bar
        cv2.putText(frame, display_text, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cooldown_remaining = max(0.0, COOLDOWN_SECONDS - (time.time() - last_trigger_time))
        cv2.putText(frame, f"Cooldown: {cooldown_remaining:.1f}s", (frame.shape[1]-200, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow("Gesture Launcher", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()

