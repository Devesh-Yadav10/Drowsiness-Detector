import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from playsound import playsound
import threading

MODEL_PATH = "models\model.h5"
model = load_model(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
blank_eye_view = np.zeros((200, 200), dtype=np.uint8)

# --- TIMERS, ALARMS & TRAINING MEMORY CHUNKS ---
CLOSED_STATE_START_TIME = None  
ALARM_TRIGGERED = False         
ALARM_THRESHOLD_SEC = 5.0       
PATH_TO_SOUND = "sounds/alarm.mp3"     

new_training_images = []
new_training_labels = []

def play_alarm_async():
    global ALARM_TRIGGERED
    try:
        playsound(PATH_TO_SOUND)
    except Exception as e:
        print(f"Audio Error: {e}")
    finally:
        ALARM_TRIGGERED = False 

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    eye_detected_this_frame = False
    current_frame_eyes_closed = False
    current_eye_processed_patch = None

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (92, 92, 140), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)
        
        if len(eyes) > 0:
            for (ex, ey, ew, eh) in eyes:
                eye_crop = roi_gray[ey:ey+eh, ex:ex+ew]
                eye_resized = cv2.resize(eye_crop, (64, 64)) 
                latest_eye_view = cv2.resize(eye_resized, (200, 200))
                eye_detected_this_frame = True
                
                # Preprocessing
                eye_normalized = eye_resized.astype("float32") / 255.0
                eye_with_channel = np.expand_dims(eye_normalized, axis=-1) 
                current_eye_processed_patch = eye_with_channel
                
                eye_input = np.expand_dims(eye_with_channel, axis=0)
                prediction = model(eye_input, training=False).numpy()
                
                if prediction[0][0] > 0.5:
                    label = "Closed"
                    color = (207, 68, 68)
                    current_frame_eyes_closed = True  
                else:
                    label = "Open"
                    color = (151, 227, 188)
                    
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
                cv2.putText(frame, label, (x + ex, y + ey - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(frame, "Eyes Closed (Lost Track)", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 243, 255), 2)
            current_frame_eyes_closed = True

    # --- TIMER & LIVE DATA AGGREGATION ---
    if current_frame_eyes_closed:
        if CLOSED_STATE_START_TIME is None:
            CLOSED_STATE_START_TIME = time.time()
        else:
            elapsed_time = time.time() - CLOSED_STATE_START_TIME
            cv2.putText(frame, f"Drowsy Warning: {elapsed_time:.1f}s", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 243, 255), 2)
            
            if elapsed_time >= ALARM_THRESHOLD_SEC:
                if not ALARM_TRIGGERED:
                    ALARM_TRIGGERED = True
                    threading.Thread(target=play_alarm_async, daemon=True).start()
                
                if current_eye_processed_patch is not None:
                    new_training_images.append(current_eye_processed_patch)
                    new_training_labels.append([0.0]) 
    else:
        CLOSED_STATE_START_TIME = None

    cv2.imshow('Eye Detection System', frame)
    
    # if eye_detected_this_frame:
    #     cv2.imshow('What the Model Sees', latest_eye_view)
    # else:
    #     cv2.putText(blank_eye_view, "Searching...", (40, 100), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    #     cv2.imshow('What the Model Sees', blank_eye_view)
    #     blank_eye_view = np.zeros((200, 200), dtype=np.uint8)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


if len(new_training_images) > 0:
    print(f"\nCollected {len(new_training_images)} new closed-eye samples from this session.")
    print("Preparing datasets for fine-tuning...")
    X_train_new = np.array(new_training_images)
    y_train_new = np.array(new_training_labels)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Tuning weights based on session metrics...")
    model.fit(X_train_new, y_train_new, epochs=5, batch_size=4, verbose=1)
    model.save(MODEL_PATH)
    print(f"Model successfully updated and saved to '{MODEL_PATH}'!")
else:
    print("\nSession ended. No new data points met the retraining criteria.")