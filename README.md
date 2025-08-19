# 🖱️ Virtual Mouse Using Hand Gestures

This project implements a **Virtual Mouse** that can be controlled entirely using hand gestures detected through a webcam. By leveraging **computer vision** and **machine learning models** for hand landmark detection, the system replaces traditional mouse input devices and enables gesture-based interaction with the computer.

---

## 📖 Description
The virtual mouse uses **OpenCV** and **MediaPipe** to detect and track hand landmarks in real time. Recognized gestures are mapped to mouse events using **PyAutoGUI**, enabling actions such as cursor movement, left/right clicks, scrolling, and drag-and-drop.  
An additional **eye-hand fusion module** allows for enhanced precision and smoother control.  

This approach demonstrates the potential of **vision-based human-computer interaction (HCI)**, offering touch-free and intuitive control mechanisms for accessibility and innovative applications.

---

## 🚀 Features
- 👆 Cursor control with index finger  
- 👉👆 Left click (index + middle finger pinch)  
- 👉💍 Right click (middle + ring finger pinch)  
- 👍👆 Scroll up/down (thumb + index distance)  
- 👍🖕 Drag & Drop (thumb + middle finger)  
- 👆🖕💍 Precision mode (stable small movements)  
- 👁️ Eye-hand fusion for better accuracy  

---

## 🛠️ Tech Stack
- Python 3  
- OpenCV  
- MediaPipe  
- PyAutoGUI  
- NumPy  

---

## ▶️ Usage
Run the scripts inside the **src/** folder:

```bash
python src/camera_test.py       # Test camera
python src/cusor_controller.py  # Main cursor control
python src/eye_hand_fussion.py  # Eye + hand fusion mode
python src/guestues.py          # Gesture recognition test

Press ESC to exit. Keep your hand inside the camera view.
