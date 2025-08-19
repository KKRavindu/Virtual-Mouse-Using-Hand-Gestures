import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time
import threading
from collections import deque

# Optimize for laptop performance
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

class LaptopCursorController:
    def __init__(self):
        # Camera setup with auto-detection
        self.cap = None
        self.camera_index = self._find_best_camera()
        
        # Screen configuration
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
        
        # MediaPipe setup - optimized for laptop cameras
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
            model_complexity=1  # Balance between speed and accuracy
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Enhanced smoothing system
        self.smoothening_factor = 8
        self.position_buffer = deque(maxlen=5)
        self.prev_x, self.prev_y = self.screen_width//2, self.screen_height//2
        
        # Gesture states
        self.gesture_states = {
            'dragging': False,
            'last_click': 0,
            'last_scroll': 0,
            'click_mode': False,
            'gesture_start_time': 0,
            'stable_gesture_time': 0.3,  # Time to hold gesture before activation
        }
        
        # Calibration zones (laptop optimized)
        self.frame_reduction = 80
        self.dead_zone = 10  # Pixels of dead zone at edges
        
        # Enhanced thresholds
        self.thresholds = {
            'click_distance': 35,
            'release_distance': 60,
            'scroll_close': 40,
            'scroll_far': 100,
            'gesture_stability': 0.2,
            'click_cooldown': 0.3,
            'scroll_cooldown': 0.1,
        }
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Gesture recognition improvements
        self.gesture_history = deque(maxlen=10)
        self.current_gesture = "None"
        
    def _find_best_camera(self):
        """Find and configure the best camera for laptop use"""
        print("🔍 Searching for laptop camera...")
        
        for index in range(3):  # Most laptops use 0-2
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                # Test camera quality
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    # Prefer higher resolution cameras
                    if width >= 640 and height >= 480:
                        # Optimize camera settings
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        print(f"✅ Using camera {index} - Resolution: {width}x{height}")
                        self.cap = cap
                        return index
                cap.release()
        
        raise Exception("❌ No suitable camera found! Please check camera permissions.")
    
    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _smooth_position(self, x, y):
        """Advanced position smoothing for stable cursor movement"""
        # Add to buffer
        self.position_buffer.append((x, y))
        
        # Weighted average with recent positions having more weight
        if len(self.position_buffer) >= 3:
            weights = [1, 2, 3]  # Recent positions weighted more
            weighted_x = sum(pos[0] * w for pos, w in zip(list(self.position_buffer)[-3:], weights))
            weighted_y = sum(pos[1] * w for pos, w in zip(list(self.position_buffer)[-3:], weights))
            total_weight = sum(weights)
            
            smooth_x = weighted_x / total_weight
            smooth_y = weighted_y / total_weight
        else:
            smooth_x, smooth_y = x, y
        
        # Apply smoothening with previous position
        final_x = self.prev_x + (smooth_x - self.prev_x) / self.smoothening_factor
        final_y = self.prev_y + (smooth_y - self.prev_y) / self.smoothening_factor
        
        self.prev_x, self.prev_y = final_x, final_y
        return int(final_x), int(final_y)
    
    def _detect_fingers_up(self, landmarks):
        """Enhanced finger detection with better accuracy"""
        fingers = []
        
        # Thumb (consider hand orientation)
        thumb_tip = landmarks[4]
        thumb_pip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        # Determine if thumb is up based on hand orientation
        if abs(thumb_tip[0] - thumb_mcp[0]) > abs(thumb_tip[1] - thumb_mcp[1]):
            # Horizontal thumb movement
            fingers.append(thumb_tip[0] > thumb_pip[0] if thumb_tip[0] > landmarks[17][0] else thumb_tip[0] < thumb_pip[0])
        else:
            # Vertical thumb movement
            fingers.append(thumb_tip[1] < thumb_pip[1])
        
        # Other fingers (improved detection)
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            # Check if fingertip is above PIP joint
            fingers.append(landmarks[tip][1] < landmarks[pip][1])
        
        return fingers
    
    def _recognize_gesture(self, fingers, landmarks):
        """Enhanced gesture recognition with stability checking"""
        current_time = time.time()
        
        # Key landmarks
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        
        gesture = "Unknown"
        action = None
        
        # 1. Cursor Movement (Index finger only)
        if fingers[1] and not any([fingers[0], fingers[2], fingers[3], fingers[4]]):
            gesture = "Cursor Movement"
            action = "move"
        
        # 2. Click Mode (Index + Middle extended)
        elif fingers[1] and fingers[2] and not any([fingers[0], fingers[3], fingers[4]]):
            distance = self._calculate_distance(index_tip, middle_tip)
            if distance < self.thresholds['click_distance']:
                gesture = "Left Click"
                action = "left_click"
            else:
                gesture = "Click Ready"
                action = "click_ready"
        
        # 3. Right Click (Middle + Ring)
        elif fingers[2] and fingers[3] and not any([fingers[0], fingers[1], fingers[4]]):
            distance = self._calculate_distance(middle_tip, ring_tip)
            if distance < self.thresholds['click_distance']:
                gesture = "Right Click"
                action = "right_click"
        
        # 4. Scroll Mode (Thumb + Index)
        elif fingers[0] and fingers[1] and not any([fingers[2], fingers[3], fingers[4]]):
            distance = self._calculate_distance(thumb_tip, index_tip)
            if distance < self.thresholds['scroll_close']:
                gesture = "Scroll Up"
                action = "scroll_up"
            elif distance > self.thresholds['scroll_far']:
                gesture = "Scroll Down"
                action = "scroll_down"
            else:
                gesture = "Scroll Ready"
        
        # 5. Drag Mode (Thumb + Middle)
        elif fingers[0] and fingers[2] and not any([fingers[1], fingers[3], fingers[4]]):
            distance = self._calculate_distance(thumb_tip, middle_tip)
            if distance < self.thresholds['click_distance']:
                gesture = "Drag Active"
                action = "drag"
            else:
                gesture = "Drag Ready"
        
        # 6. Precision Mode (All fingers except pinky)
        elif fingers[1] and fingers[2] and fingers[3] and not fingers[4]:
            gesture = "Precision Mode"
            action = "precision"
        
        # Add to gesture history for stability
        self.gesture_history.append((gesture, action))
        
        # Check for stable gesture
        if len(self.gesture_history) >= 5:
            recent_gestures = [g[0] for g in list(self.gesture_history)[-5:]]
            if len(set(recent_gestures)) == 1:  # All same gesture
                stable_gesture = recent_gestures[0]
                stable_action = self.gesture_history[-1][1]
                return stable_gesture, stable_action
        
        return gesture, action
    
    def _execute_action(self, action, landmarks):
        """Execute the recognized action with improved accuracy"""
        current_time = time.time()
        
        if action == "move":
            index_tip = landmarks[8]
            self._move_cursor(index_tip)
            
        elif action == "left_click":
            if current_time - self.gesture_states['last_click'] > self.thresholds['click_cooldown']:
                pyautogui.click()
                self.gesture_states['last_click'] = current_time
                print("🖱️ Left Click")
                
        elif action == "right_click":
            if current_time - self.gesture_states['last_click'] > self.thresholds['click_cooldown']:
                pyautogui.click(button='right')
                self.gesture_states['last_click'] = current_time
                print("🖱️ Right Click")
                
        elif action == "scroll_up":
            if current_time - self.gesture_states['last_scroll'] > self.thresholds['scroll_cooldown']:
                pyautogui.scroll(3)
                self.gesture_states['last_scroll'] = current_time
                
        elif action == "scroll_down":
            if current_time - self.gesture_states['last_scroll'] > self.thresholds['scroll_cooldown']:
                pyautogui.scroll(-3)
                self.gesture_states['last_scroll'] = current_time
                
        elif action == "drag":
            if not self.gesture_states['dragging']:
                pyautogui.mouseDown()
                self.gesture_states['dragging'] = True
                print("🖱️ Drag Started")
            # Continue moving cursor while dragging
            index_tip = landmarks[8]
            self._move_cursor(index_tip)
            
        elif action == "precision":
            # Precision mode with reduced sensitivity
            index_tip = landmarks[8]
            self._move_cursor(index_tip, precision=True)
        
        # Stop dragging if not in drag mode
        if action != "drag" and self.gesture_states['dragging']:
            pyautogui.mouseUp()
            self.gesture_states['dragging'] = False
            print("🖱️ Drag Ended")
    
    def _move_cursor(self, finger_pos, precision=False):
        """Enhanced cursor movement with precision mode"""
        x, y = finger_pos
        
        # Map finger position to screen coordinates
        screen_x = np.interp(x, 
                           (self.frame_reduction, self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) - self.frame_reduction),
                           (0, self.screen_width))
        screen_y = np.interp(y, 
                           (self.frame_reduction, self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - self.frame_reduction),
                           (0, self.screen_height))
        
        # Apply precision mode (reduce sensitivity)
        if precision:
            current_x, current_y = pyautogui.position()
            screen_x = current_x + (screen_x - self.screen_width/2) * 0.3
            screen_y = current_y + (screen_y - self.screen_height/2) * 0.3
        
        # Smooth the movement
        final_x, final_y = self._smooth_position(screen_x, screen_y)
        
        # Apply bounds checking with dead zone
        final_x = max(self.dead_zone, min(self.screen_width - self.dead_zone, final_x))
        final_y = max(self.dead_zone, min(self.screen_height - self.dead_zone, final_y))
        
        # Move cursor
        pyautogui.moveTo(final_x, final_y)
    
    def _draw_ui(self, frame, gesture, fingers, landmarks=None):
        """Enhanced UI with more information and visual feedback"""
        h, w = frame.shape[:2]
        
        # Control area
        cv2.rectangle(frame, (self.frame_reduction, self.frame_reduction),
                     (w - self.frame_reduction, h - self.frame_reduction), (255, 0, 255), 2)
        
        # FPS Counter
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1:
            fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
            cv2.putText(frame, f"FPS: {fps}", (w-100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Current gesture with background
        gesture_bg_color = (0, 0, 0)
        if "Click" in gesture:
            gesture_bg_color = (0, 255, 0)
        elif "Scroll" in gesture:
            gesture_bg_color = (255, 255, 0)
        elif "Drag" in gesture:
            gesture_bg_color = (0, 0, 255)
        
        cv2.rectangle(frame, (10, h-80), (300, h-20), gesture_bg_color, -1)
        cv2.putText(frame, f"Gesture: {gesture}", (15, h-50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Finger status
        finger_names = ["👍", "👆", "🖕", "💍", "🤙"]
        for i, (name, up) in enumerate(zip(finger_names, fingers)):
            color = (0, 255, 0) if up else (100, 100, 100)
            cv2.putText(frame, f"{name}", (10 + i*40, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Visual feedback for gestures
        if landmarks and gesture != "Unknown":
            if "Click" in gesture:
                cv2.circle(frame, landmarks[8], 20, (0, 255, 0), 3)
            elif "Scroll" in gesture:
                cv2.line(frame, landmarks[4], landmarks[8], (255, 255, 0), 3)
            elif "Drag" in gesture:
                cv2.circle(frame, landmarks[12], 25, (0, 0, 255), 3)
        
        # Instructions
        instructions = [
            "👆 Index only: Move cursor",
            "👆+🖕 close: Left click",
            "🖕+💍 close: Right click", 
            "👍+👆: Scroll (close=up, far=down)",
            "👍+🖕 close: Drag & drop",
            "👆🖕💍: Precision mode",
            "ESC: Exit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (w-350, 60 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self):
        """Main execution loop with enhanced error handling"""
        print("\n🚀 Advanced Laptop Cursor Controller Started!")
        print("📹 Position your hand in the purple rectangle")
        print("💡 Ensure good lighting and plain background")
        print("⚡ Press ESC to exit\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Camera feed lost!")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # Default values
                gesture = "No Hand Detected"
                fingers = [False] * 5
                
                # Process hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get landmark positions
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            landmarks.append((cx, cy))
                        
                        # Draw hand landmarks
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, 
                                                   self.mp_hands.HAND_CONNECTIONS)
                        
                        # Detect fingers and gestures
                        fingers = self._detect_fingers_up(landmarks)
                        gesture, action = self._recognize_gesture(fingers, landmarks)
                        
                        # Execute action if hand is in control area
                        index_tip = landmarks[8]
                        if (self.frame_reduction < index_tip[0] < w - self.frame_reduction and 
                            self.frame_reduction < index_tip[1] < h - self.frame_reduction):
                            if action:
                                self._execute_action(action, landmarks)
                        
                        # Update current gesture for UI
                        self.current_gesture = gesture
                        break
                
                # Draw UI
                self._draw_ui(frame, gesture, fingers, 
                             landmarks if 'landmarks' in locals() else None)
                
                # Show frame
                cv2.imshow("Advanced Laptop Cursor Controller", frame)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
                elif key == ord('c'):  # 'c' key for calibration reset
                    self.position_buffer.clear()
                    print("📐 Position calibration reset")
                
        except KeyboardInterrupt:
            print("\n⚠️ Program interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources"""
        if self.gesture_states['dragging']:
            pyautogui.mouseUp()
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Laptop Cursor Controller stopped safely!")

# Run the enhanced controller
if __name__ == "__main__":
    try:
        controller = LaptopCursorController()
        controller.run()
    except Exception as e:
        print(f"Failed to start: {e}")
        print("\nTroubleshooting tips:")
        print("1. Close other apps using the camera")
        print("2. Check camera permissions")
        print("3. Ensure good lighting")
        print("4. Try running as administrator")