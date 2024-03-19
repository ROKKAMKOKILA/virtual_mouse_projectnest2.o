import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_x, index_y = 0, 0
thumb_x, thumb_y = 0, 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                if id == 8:  # Index finger
                    index_x = screen_width * landmark.x
                    index_y = screen_height * landmark.y
                elif id == 4:  # Thumb
                    thumb_x = screen_width * landmark.x
                    thumb_y = screen_height * landmark.y
                    cv2.line(frame, (int(index_x), int(index_y)), (int(thumb_x), int(thumb_y)), (0, 255, 0),
                             2)  # Green line between index and thumb

                    if abs(index_y - thumb_y) < 20:
                        pyautogui.click()
                        pyautogui.sleep(0.5)
                    elif abs(index_y - thumb_y) < 100:
                        pyautogui.moveTo(index_x, index_y, duration=0.2)  # Slower cursor movement

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()