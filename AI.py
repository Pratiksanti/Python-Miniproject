import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

tip_ids = [4, 8, 12, 16, 20]
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    lm_list = []
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

    fingers = []
    gesture = ""

    if lm_list:
       
        fingers.append(1 if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1] else 0)
        for i in range(1, 5):
            fingers.append(1 if lm_list[tip_ids[i]][2] < lm_list[tip_ids[i] - 2][2] else 0)
        finger_map = {
            (0,0,0,0,0): "Fist 👊",
            (0,1,0,0,0): "One ☝️",
            (0,1,1,0,0): "Peace ✌️",
            (0,1,1,1,0): "All the Best 🤞",
            (1,0,0,0,0): "Super 👍",
            (1,1,0,0,0): "Like & Point 👉👍",
            (1,1,1,0,0): "Good Job 👌",
            (1,1,1,1,0): "Rock 🤘",
            (1,1,1,1,1): "Hi 👋",
            (0,0,0,0,1): "Pinky Promise 🩷",
        }

        gesture = finger_map.get(tuple(fingers), "Unknown 🤔")
        
        cv2.rectangle(img, (20, 250), (500, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, gesture, (30, 360), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (0, 0, 255), 4)

    cv2.imshow("Hand Gesture Emoji", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
