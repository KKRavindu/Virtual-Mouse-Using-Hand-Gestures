import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time

# Disable PyAutoGUI fail-safe for better performance
pyautogui.FAILSAFE = False

# Initialize camera and screen size
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# Variables for smoothing and control
prev_x, prev_y = 0, 0
smoothening = 7
frame_reduction = 100  # Reduce frame area for better control

# Action states
dragging = False
last_click_time = 0
last_scroll_time = 0
click_cooldown = 0.3
scroll_cooldown = 0.1

# Gesture thresholds
PINCH_THRESHOLD = 40
RELEASE_THRESHOLD = 70
SCROLL_NEAR = 50
SCROLL_FAR = 120

def fingers_up(lm_list):
    """Detect which fingers are up"""
    fingers = []
    
    # Thumb (check x-coordinate for left/right hand detection)
    if lm_list[4][0] > lm_list[3][0]:  # Right hand
        fingers.append(lm_list[4][0] > lm_list[3][0])
    else:  # Left hand
        fingers.append(lm_list[4][0] < lm_list[3][0])
    
    # Other fingers (y-coordinate comparison)
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    
    for tip, pip in zip(tip_ids, pip_ids):
        fingers.append(lm_list[tip][1] < lm_list[pip][1])
    
    return fingers

def get_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

def draw_gesture_info(frame, fingers, gesture_text=""):
    """Draw finger status and current gesture"""
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    
    # Draw finger status
    for i, (name, up) in enumerate(zip(finger_names, fingers)):
        color = (0, 255, 0) if up else (0, 0, 255)
        cv2.putText(frame, f"{name}: {'UP' if up else 'DOWN'}", 
                   (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw current gesture
    if gesture_text:
        cv2.putText(frame, f"Gesture: {gesture_text}", 
                   (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

print("Virtual Mouse Controller Started!")
print("Gestures:")
print("- Index finger only: Move cursor")
print("- Index + Middle pinch: Left click") 
print("- Middle + Ring pinch: Right click")
print("- Thumb + Index close: Scroll up")
print("- Thumb + Index far: Scroll down")
print("- Thumb + Index pinch & hold: Drag")
print("- Press ESC to exit")

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Create a control area (reduce jitter)
        cv2.rectangle(frame, (frame_reduction, frame_reduction), 
                     (w - frame_reduction, h - frame_reduction), (255, 0, 255), 2)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        current_time = time.time()
        gesture_text = ""

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Get landmark positions
                lm_list = []
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy))

                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if lm_list:
                    fingers = fingers_up(lm_list)
                    
                    # Key landmark points
                    thumb_tip = lm_list[4]
                    index_tip = lm_list[8]
                    middle_tip = lm_list[12]
                    ring_tip = lm_list[16]
                    
                    # Only process if hand is in control area
                    if (frame_reduction < index_tip[0] < w - frame_reduction and 
                        frame_reduction < index_tip[1] < h - frame_reduction):
                        
                        # 1. Cursor Movement (Index finger only)
                        if fingers[1] and not any(fingers[2:]) and not fingers[0]:
                            screen_x = np.interp(index_tip[0], 
                                               (frame_reduction, w - frame_reduction), 
                                               (0, screen_width))
                            screen_y = np.interp(index_tip[1], 
                                               (frame_reduction, h - frame_reduction), 
                                               (0, screen_height))
                            
                            # Smooth movement
                            curr_x = prev_x + (screen_x - prev_x) / smoothening
                            curr_y = prev_y + (screen_y - prev_y) / smoothening
                            
                            pyautogui.moveTo(curr_x, curr_y)
                            prev_x, prev_y = curr_x, curr_y
                            gesture_text = "Moving Cursor"
                            
                            # Reset drag if active
                            if dragging:
                                pyautogui.mouseUp()
                                dragging = False

                        # 2. Left Click (Index + Middle pinch)
                        elif fingers[1] and fingers[2] and not any(fingers[3:]) and not fingers[0]:
                            dist = get_distance(index_tip, middle_tip)
                            if (dist < PINCH_THRESHOLD and 
                                current_time - last_click_time > click_cooldown):
                                pyautogui.click()
                                last_click_time = current_time
                                gesture_text = "Left Click"
                                cv2.circle(frame, index_tip, 15, (0, 255, 0), -1)

                        # 3. Right Click (Middle + Ring pinch)
                        elif fingers[2] and fingers[3] and not fingers[1] and not fingers[4] and not fingers[0]:
                            dist = get_distance(middle_tip, ring_tip)
                            if (dist < PINCH_THRESHOLD and 
                                current_time - last_click_time > click_cooldown):
                                pyautogui.click(button='right')
                                last_click_time = current_time
                                gesture_text = "Right Click"
                                cv2.circle(frame, middle_tip, 15, (0, 0, 255), -1)

                        # 4. Scroll (Thumb + Index)
                        elif fingers[0] and fingers[1] and not any(fingers[2:]):
                            dist = get_distance(thumb_tip, index_tip)
                            if current_time - last_scroll_time > scroll_cooldown:
                                if dist < SCROLL_NEAR:
                                    pyautogui.scroll(3)  # Scroll up
                                    gesture_text = "Scroll Up"
                                    last_scroll_time = current_time
                                elif dist > SCROLL_FAR:
                                    pyautogui.scroll(-3)  # Scroll down
                                    gesture_text = "Scroll Down"
                                    last_scroll_time = current_time
                            
                            # Visual feedback
                            cv2.line(frame, thumb_tip, index_tip, (255, 255, 0), 3)

                        # 5. Drag and Drop (Thumb + Index + Middle)
                        elif fingers[0] and fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                            dist = get_distance(thumb_tip, index_tip)
                            
                            if dist < PINCH_THRESHOLD and not dragging:
                                pyautogui.mouseDown()
                                dragging = True
                                gesture_text = "Start Drag"
                            elif dist > RELEASE_THRESHOLD and dragging:
                                pyautogui.mouseUp()
                                dragging = False
                                gesture_text = "End Drag"
                            elif dragging:
                                gesture_text = "Dragging"
                                # Move cursor while dragging
                                screen_x = np.interp(index_tip[0], 
                                                   (frame_reduction, w - frame_reduction), 
                                                   (0, screen_width))
                                screen_y = np.interp(index_tip[1], 
                                                   (frame_reduction, h - frame_reduction), 
                                                   (0, screen_height))
                                curr_x = prev_x + (screen_x - prev_x) / smoothening
                                curr_y = prev_y + (screen_y - prev_y) / smoothening
                                pyautogui.moveTo(curr_x, curr_y)
                                prev_x, prev_y = curr_x, curr_y
                            
                            # Visual feedback for drag
                            if dragging:
                                cv2.circle(frame, index_tip, 20, (0, 255, 255), 3)

                    # Draw gesture information
                    draw_gesture_info(frame, fingers, gesture_text)

        # Instructions
        cv2.putText(frame, "ESC to exit", (w-120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Virtual Mouse Controller", frame)
        
        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Cleanup
    if dragging:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()
    print("Virtual Mouse Controller stopped.")