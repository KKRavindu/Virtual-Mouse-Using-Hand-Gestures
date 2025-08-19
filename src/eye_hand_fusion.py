import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time
import threading
from collections import deque
from scipy.spatial import distance as dist

# Optimize for laptop performance
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

class EyeHandFusionController:
    def __init__(self):
        # Camera setup with auto-detection
        self.cap = None
        self.camera_index = self._find_best_camera()
        
        # Screen configuration
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
        
        # MediaPipe setup - Face and Hand detection
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
            model_complexity=1
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye tracking setup
        self.left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Eye aspect ratio for blink detection
        self.ear_threshold = 0.25
        self.ear_consec_frames = 3
        self.ear_counter = 0
        self.total_blinks = 0
        
        # Gaze tracking
        self.gaze_history = deque(maxlen=10)
        self.stable_gaze_threshold = 0.7
        self.gaze_click_enabled = False
        
        # Enhanced smoothing system
        self.smoothening_factor = 6
        self.position_buffer = deque(maxlen=8)
        self.prev_x, self.prev_y = self.screen_width//2, self.screen_height//2
        
        # Eye-hand coordination states
        self.control_mode = "hand"  # "hand", "eye", "fusion"
        self.eye_cursor_active = False
        self.last_mode_switch = time.time()
        self.mode_cooldown = 2.0
        
        # Gesture states with eye integration
        self.gesture_states = {
            'dragging': False,
            'last_click': 0,
            'last_scroll': 0,
            'last_blink_click': 0,
            'gesture_start_time': 0,
            'stable_gesture_time': 0.2,
            'eye_dwell_start': 0,
            'dwell_click_time': 1.5,  # Seconds to dwell for click
        }
        
        # Calibration zones
        self.frame_reduction = 60
        self.dead_zone = 15
        
        # Enhanced thresholds
        self.thresholds = {
            'click_distance': 30,
            'release_distance': 55,
            'scroll_close': 35,
            'scroll_far': 90,
            'blink_click_cooldown': 1.0,
            'dwell_tolerance': 50,  # Pixels
            'eye_stability_frames': 5,
        }
        
        # Performance and UI
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.gesture_history = deque(maxlen=8)
        self.current_gesture = "Initializing"
        
        # Eye tracking calibration
        self.eye_calibrated = False
        self.calibration_points = []
        self.calibration_phase = 0
        
        # Dwell click functionality
        self.dwell_position = None
        self.dwell_start_time = 0
        self.dwell_circle_progress = 0
        
    def _find_best_camera(self):
        """Find and configure the best camera for laptop use"""
        print("🔍 Searching for laptop camera with face detection capability...")
        
        for index in range(3):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    if width >= 640 and height >= 480:
                        # Optimize for face and hand detection
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                        
                        print(f"✅ Camera {index} configured for eye-hand tracking")
                        self.cap = cap
                        return index
                cap.release()
        
        raise Exception("❌ No suitable camera found!")
    
    def _calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio for blink detection"""
        # Vertical eye landmarks
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        # Horizontal eye landmark
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def _extract_eye_landmarks(self, landmarks, eye_indices):
        """Extract eye landmarks from face mesh"""
        eye_points = []
        for idx in eye_indices[:6]:  # Take first 6 points for EAR calculation
            point = landmarks[idx]
            eye_points.append([point.x, point.y])
        return np.array(eye_points)
    
    def _detect_blink(self, face_landmarks):
        """Detect blinks using both eyes"""
        if not face_landmarks:
            return False
        
        # Extract eye landmarks
        left_eye = self._extract_eye_landmarks(face_landmarks, self.left_eye_indices)
        right_eye = self._extract_eye_landmarks(face_landmarks, self.right_eye_indices)
        
        # Calculate EAR for both eyes
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Check if blink detected
        if avg_ear < self.ear_threshold:
            self.ear_counter += 1
        else:
            if self.ear_counter >= self.ear_consec_frames:
                self.total_blinks += 1
                self.ear_counter = 0
                return True
            self.ear_counter = 0
        
        return False
    
    def _estimate_gaze_direction(self, face_landmarks, frame_shape):
        """Estimate gaze direction from face landmarks"""
        if not face_landmarks:
            return None, None
        
        h, w = frame_shape[:2]
        
        # Key landmarks for gaze estimation
        nose_tip = face_landmarks[1]  # Nose tip
        left_eye_center = face_landmarks[468]  # Left eye center
        right_eye_center = face_landmarks[473]  # Right eye center
        
        # Convert to pixel coordinates
        nose_x = int(nose_tip.x * w)
        nose_y = int(nose_tip.y * h)
        left_eye_x = int(left_eye_center.x * w)
        left_eye_y = int(left_eye_center.y * h)
        right_eye_x = int(right_eye_center.x * w)
        right_eye_y = int(right_eye_center.y * h)
        
        # Calculate eye center
        eye_center_x = (left_eye_x + right_eye_x) // 2
        eye_center_y = (left_eye_y + right_eye_y) // 2
        
        # Estimate gaze direction relative to nose
        gaze_x = eye_center_x - nose_x
        gaze_y = eye_center_y - nose_y
        
        # Map to screen coordinates (simplified mapping)
        screen_x = np.interp(-gaze_x, (-100, 100), (0, self.screen_width))
        screen_y = np.interp(-gaze_y, (-100, 100), (0, self.screen_height))
        
        return screen_x, screen_y
    
    def _update_control_mode(self, hand_detected, face_detected, fingers):
        """Intelligently switch between control modes based on context"""
        current_time = time.time()
        
        if current_time - self.last_mode_switch < self.mode_cooldown:
            return
        
        # Mode switching logic
        if hand_detected and face_detected:
            # Both available - use gesture to determine mode
            if sum(fingers) == 0:  # Closed fist = eye mode
                if self.control_mode != "eye":
                    self.control_mode = "eye"
                    self.eye_cursor_active = True
                    self.last_mode_switch = current_time
                    print("👁️ Switched to EYE CONTROL mode")
            elif sum(fingers) >= 3:  # Open hand = fusion mode
                if self.control_mode != "fusion":
                    self.control_mode = "fusion"
                    self.last_mode_switch = current_time
                    print("🤝 Switched to FUSION mode (Eye + Hand)")
            else:  # Normal gestures = hand mode
                if self.control_mode != "hand":
                    self.control_mode = "hand"
                    self.eye_cursor_active = False
                    self.last_mode_switch = current_time
                    print("✋ Switched to HAND CONTROL mode")
        elif face_detected and not hand_detected:
            # Only face - auto switch to eye mode
            if self.control_mode != "eye":
                self.control_mode = "eye"
                self.eye_cursor_active = True
                self.last_mode_switch = current_time
                print("👁️ Auto-switched to EYE CONTROL (no hand detected)")
        elif hand_detected and not face_detected:
            # Only hand - hand mode
            if self.control_mode != "hand":
                self.control_mode = "hand"
                self.eye_cursor_active = False
                self.last_mode_switch = current_time
    
    def _process_eye_control(self, gaze_x, gaze_y):
        """Process eye-based cursor control with dwell clicking"""
        if gaze_x is None or gaze_y is None:
            return
        
        current_time = time.time()
        
        # Smooth gaze movement
        if len(self.gaze_history) > 0:
            prev_gaze = self.gaze_history[-1]
            smooth_x = prev_gaze[0] + (gaze_x - prev_gaze[0]) * 0.3
            smooth_y = prev_gaze[1] + (gaze_y - prev_gaze[1]) * 0.3
        else:
            smooth_x, smooth_y = gaze_x, gaze_y
        
        self.gaze_history.append((smooth_x, smooth_y))
        
        # Move cursor with gaze
        final_x = max(self.dead_zone, min(self.screen_width - self.dead_zone, smooth_x))
        final_y = max(self.dead_zone, min(self.screen_height - self.dead_zone, smooth_y))
        
        pyautogui.moveTo(final_x, final_y)
        
        # Dwell clicking - click if gaze stays in same area
        current_pos = pyautogui.position()
        
        if self.dwell_position is None:
            self.dwell_position = current_pos
            self.dwell_start_time = current_time
        else:
            distance_moved = math.sqrt((current_pos[0] - self.dwell_position[0])**2 + 
                                     (current_pos[1] - self.dwell_position[1])**2)
            
            if distance_moved < self.thresholds['dwell_tolerance']:
                # Still dwelling
                dwell_duration = current_time - self.dwell_start_time
                self.dwell_circle_progress = min(1.0, dwell_duration / self.gesture_states['dwell_click_time'])
                
                if dwell_duration >= self.gesture_states['dwell_click_time']:
                    # Dwell click!
                    pyautogui.click()
                    print("👁️ Dwell Click!")
                    self.dwell_position = None
                    self.dwell_circle_progress = 0
            else:
                # Moved too much, reset dwell
                self.dwell_position = current_pos
                self.dwell_start_time = current_time
                self.dwell_circle_progress = 0
    
    def _detect_fingers_up(self, landmarks):
        """Enhanced finger detection"""
        fingers = []
        
        # Thumb
        if landmarks[4][0] > landmarks[3][0]:
            fingers.append(landmarks[4][0] > landmarks[3][0])
        else:
            fingers.append(landmarks[4][0] < landmarks[3][0])
        
        # Other fingers
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            fingers.append(landmarks[tip][1] < landmarks[pip][1])
        
        return fingers
    
    def _recognize_gesture(self, fingers, landmarks, control_mode):
        """Enhanced gesture recognition with mode awareness"""
        if control_mode == "eye":
            return "Eye Control Active", "eye_control"
        
        gesture = "Unknown"
        action = None
        
        # Standard hand gestures
        if fingers[1] and not any([fingers[0], fingers[2], fingers[3], fingers[4]]):
            gesture = "Cursor Movement"
            action = "move"
        elif fingers[1] and fingers[2] and not any([fingers[0], fingers[3], fingers[4]]):
            distance = self._calculate_distance(landmarks[8], landmarks[12])
            if distance < self.thresholds['click_distance']:
                gesture = "Left Click"
                action = "left_click"
        elif fingers[2] and fingers[3] and not any([fingers[0], fingers[1], fingers[4]]):
            distance = self._calculate_distance(landmarks[12], landmarks[16])
            if distance < self.thresholds['click_distance']:
                gesture = "Right Click"
                action = "right_click"
        elif fingers[0] and fingers[1] and not any([fingers[2], fingers[3], fingers[4]]):
            distance = self._calculate_distance(landmarks[4], landmarks[8])
            if distance < self.thresholds['scroll_close']:
                gesture = "Scroll Up"
                action = "scroll_up"
            elif distance > self.thresholds['scroll_far']:
                gesture = "Scroll Down" 
                action = "scroll_down"
        elif sum(fingers) == 0:
            gesture = "Switch to Eye Mode"
            action = "switch_eye"
        elif sum(fingers) >= 4:
            gesture = "Fusion Mode"
            action = "fusion"
        
        return gesture, action
    
    def _calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _execute_action(self, action, landmarks=None, blink_detected=False):
        """Execute actions with eye-hand coordination"""
        current_time = time.time()
        
        if action == "move" and landmarks:
            index_tip = landmarks[8]
            self._move_cursor_hand(index_tip)
            
        elif action == "left_click":
            if current_time - self.gesture_states['last_click'] > 0.3:
                pyautogui.click()
                self.gesture_states['last_click'] = current_time
                print("🖱️ Hand Left Click")
                
        elif action == "right_click":
            if current_time - self.gesture_states['last_click'] > 0.3:
                pyautogui.click(button='right')
                self.gesture_states['last_click'] = current_time
                print("🖱️ Hand Right Click")
                
        elif action == "scroll_up":
            pyautogui.scroll(3)
        elif action == "scroll_down":
            pyautogui.scroll(-3)
            
        elif action == "eye_control":
            # Eye control is handled in _process_eye_control
            pass
        
        # Blink-based actions (works in all modes)
        if (blink_detected and 
            current_time - self.gesture_states['last_blink_click'] > self.thresholds['blink_click_cooldown']):
            if self.control_mode == "eye":
                # In eye mode, blink = click
                pyautogui.click()
                print("👁️ Blink Click!")
                self.gesture_states['last_blink_click'] = current_time
            elif self.control_mode == "fusion":
                # In fusion mode, blink = right click
                pyautogui.click(button='right')
                print("👁️ Blink Right Click!")
                self.gesture_states['last_blink_click'] = current_time
    
    def _move_cursor_hand(self, finger_pos):
        """Hand-based cursor movement"""
        x, y = finger_pos
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        screen_x = np.interp(x, (self.frame_reduction, w - self.frame_reduction), (0, self.screen_width))
        screen_y = np.interp(y, (self.frame_reduction, h - self.frame_reduction), (0, self.screen_height))
        
        # Smooth movement
        final_x = self.prev_x + (screen_x - self.prev_x) / self.smoothening_factor
        final_y = self.prev_y + (screen_y - self.prev_y) / self.smoothening_factor
        
        self.prev_x, self.prev_y = final_x, final_y
        pyautogui.moveTo(final_x, final_y)
    
    def _draw_enhanced_ui(self, frame, gesture, fingers, control_mode, blink_detected=False):
        """Enhanced UI with eye tracking information"""
        h, w = frame.shape[:2]
        
        # Control area
        cv2.rectangle(frame, (self.frame_reduction, self.frame_reduction),
                     (w - self.frame_reduction, h - self.frame_reduction), (255, 0, 255), 2)
        
        # Control mode indicator
        mode_colors = {"hand": (0, 255, 0), "eye": (255, 0, 0), "fusion": (0, 255, 255)}
        mode_color = mode_colors.get(control_mode, (255, 255, 255))
        
        cv2.rectangle(frame, (10, 10), (250, 60), mode_color, -1)
        cv2.putText(frame, f"Mode: {control_mode.upper()}", (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Blink indicator
        if blink_detected:
            cv2.circle(frame, (w-50, 50), 20, (0, 255, 0), -1)
            cv2.putText(frame, "BLINK", (w-80, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Dwell click progress (eye mode)
        if control_mode == "eye" and self.dwell_circle_progress > 0:
            center = (w//2, h//2)
            radius = 40
            angle = int(360 * self.dwell_circle_progress)
            cv2.ellipse(frame, center, (radius, radius), 0, 0, angle, (0, 255, 255), 5)
            if self.dwell_circle_progress >= 1.0:
                cv2.circle(frame, center, radius, (0, 255, 0), 3)
        
        # Gesture info
        cv2.rectangle(frame, (10, h-100), (400, h-20), (0, 0, 0), -1)
        cv2.putText(frame, f"Gesture: {gesture}", (15, h-70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {self.total_blinks}", (15, h-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions based on mode
        instructions = []
        if control_mode == "hand":
            instructions = [
                "✋ HAND MODE - Fist: Switch to Eye",
                "👆 Index: Move | 👆🖕: Click",
                "👍👆: Scroll | Open Hand: Fusion"
            ]
        elif control_mode == "eye":
            instructions = [
                "👁️ EYE MODE - Look to move cursor",
                "Dwell 1.5s to click | Blink: Click",
                "Hand gestures: Switch modes"
            ]
        elif control_mode == "fusion":
            instructions = [
                "🤝 FUSION MODE - Eye + Hand",
                "Eye: Move | Hand: Actions",
                "Blink: Right Click"
            ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (w-450, 100 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self):
        """Main execution loop with eye-hand fusion"""
        print("\n🚀 Eye-Hand Fusion Cursor Controller Started!")
        print("👁️ Look around to move cursor in Eye Mode")
        print("✋ Use hand gestures for precise control")
        print("🤝 Fusion mode combines both!")
        print("⚡ Press ESC to exit\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process both face and hand
                face_results = self.face_mesh.process(rgb_frame)
                hand_results = self.hands.process(rgb_frame)
                
                # Default values
                gesture = "Waiting..."
                fingers = [False] * 5
                hand_detected = False
                face_detected = False
                blink_detected = False
                gaze_x, gaze_y = None, None
                
                # Process face landmarks
                if face_results.multi_face_landmarks:
                    face_detected = True
                    face_landmarks = face_results.multi_face_landmarks[0].landmark
                    
                    # Detect blinks
                    blink_detected = self._detect_blink(face_landmarks)
                    
                    # Estimate gaze
                    gaze_x, gaze_y = self._estimate_gaze_direction(face_landmarks, frame.shape)
                    
                    # Draw face landmarks (simplified)
                    self.mp_drawing.draw_landmarks(
                        frame, face_results.multi_face_landmarks[0], 
                        self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)
                    )
                
                # Process hand landmarks
                landmarks = None
                if hand_results.multi_hand_landmarks:
                    hand_detected = True
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # Get landmark positions
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            landmarks.append((cx, cy))
                        
                        # Draw hand landmarks
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, 
                                                     self.mp_hands.HAND_CONNECTIONS)
                        
                        # Detect fingers
                        fingers = self._detect_fingers_up(landmarks)
                        break
                
                # Update control mode based on context
                self._update_control_mode(hand_detected, face_detected, fingers)
                
                # Execute control based on current mode
                if self.control_mode == "eye" and face_detected:
                    gesture = "Eye Control Active"
                    self._process_eye_control(gaze_x, gaze_y)
                    
                elif self.control_mode == "hand" and hand_detected:
                    gesture, action = self._recognize_gesture(fingers, landmarks, self.control_mode)
                    if action:
                        self._execute_action(action, landmarks)
                        
                elif self.control_mode == "fusion":
                    if face_detected:
                        self._process_eye_control(gaze_x, gaze_y)
                    if hand_detected:
                        gesture, action = self._recognize_gesture(fingers, landmarks, self.control_mode)
                        if action and action not in ["move"]:  # Eye handles movement in fusion
                            self._execute_action(action, landmarks)
                    gesture = "Fusion: Eye+Hand Active"
                
                # Handle blink actions
                if blink_detected:
                    self._execute_action("blink", blink_detected=True)
                
                # Draw enhanced UI
                self._draw_enhanced_ui(frame, gesture, fingers, self.control_mode, blink_detected)
                
                cv2.imshow("Eye-Hand Fusion Cursor Controller", frame)
                
                # Exit on ESC
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == ord('1'):
                    self.control_mode = "hand"
                    print("Switched to Hand Mode")
                elif key == ord('2'):
                    self.control_mode = "eye" 
                    print("Switched to Eye Mode")
                elif key == ord('3'):
                    self.control_mode = "fusion"
                    print("Switched to Fusion Mode")
                
        except KeyboardInterrupt:
            print("\n⚠️ Program interrupted")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Eye-Hand Fusion Controller stopped!")

# Run the enhanced eye-hand fusion controller
if __name__ == "__main__":
    try:
        controller = EyeHandFusionController()
        controller.run()
    except Exception as e:
        print(f"Failed to start: {e}")
        print("Make sure you have good lighting for face detection!")