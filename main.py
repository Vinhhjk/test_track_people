import cv2
import time
import numpy as np
import threading
from queue import Queue
import face_recognition
import sqlite3
from datetime import datetime
import time 

from config import CAMERA_URL, CACHE_UPDATE_INTERVAL, ABSENCE_THRESHOLD, MODEL_PATHS, age_list, gender_list

cap = cv2.VideoCapture(CAMERA_URL) 

# Load OpenCV's pre-trained deep learning models for Age & Gender detection

face_net = cv2.dnn.readNetFromCaffe(*MODEL_PATHS['face_detector'])
age_net = cv2.dnn.readNetFromCaffe(*MODEL_PATHS['age_net'])
gender_net = cv2.dnn.readNetFromCaffe(*MODEL_PATHS['gender_net'])

frame_queue = Queue(maxsize=2)
results_queue = Queue(maxsize=2)

# Queues for thread communication
frame_queue = Queue(maxsize=2)
results_queue = Queue(maxsize=2)

KNOWN_ENCODINGS = []
KNOWN_IDS = []
CUSTOMER_INFO = {}
LAST_CACHE_UPDATE = 0
CACHE_UPDATE_INTERVAL = 30  # Update cache every 30 seconds

def update_customer_cache():
    global KNOWN_ENCODINGS, KNOWN_IDS, CUSTOMER_INFO, LAST_CACHE_UPDATE
    current_time = time.time()
    
    if current_time - LAST_CACHE_UPDATE >= CACHE_UPDATE_INTERVAL:
        KNOWN_ENCODINGS, KNOWN_IDS, _, _, CUSTOMER_INFO = get_known_customers()
        LAST_CACHE_UPDATE = current_time

def get_known_customers():
    with sqlite3.connect('customer_data.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, face_encoding, visit_count, last_location, age, gender FROM customers")
        customers = cursor.fetchall()
        known_encodings = []
        known_ids = []
        visit_counts = {}
        last_locations = {}
        customer_info = {}
        
        for customer in customers:
            customer_id, encoding_str, visit_count, last_location, age, gender = customer
            encoding = np.fromstring(encoding_str[1:-1], sep=',')
            known_encodings.append(encoding)
            known_ids.append(customer_id)
            visit_counts[customer_id] = visit_count
            last_locations[customer_id] = last_location
            customer_info[customer_id] = {'age': age, 'gender': gender}
        
        return known_encodings, known_ids, visit_counts, last_locations, customer_info

CUSTOMERS_IN_FRAME = {}
LAST_SEEN_TIMESTAMP = {}
ABSENCE_THRESHOLD = 10  

def update_customer(customer_id, location):
    with sqlite3.connect('customer_data.db') as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE customers SET visit_count = visit_count + 1, last_seen = ?, last_location = ? WHERE id = ?", 
                      (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), location, customer_id))
        conn.commit()


def add_new_customer(face_encoding, location, age, gender):
    with sqlite3.connect('customer_data.db') as conn:
        cursor = conn.cursor()
        encoding_str = str(list(face_encoding))
        cursor.execute("""
            INSERT INTO customers 
            (face_encoding, visit_count, last_seen, last_location, age, gender) 
            VALUES (?, ?, ?, ?, ?, ?)""",
            (encoding_str, 1, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
             location, age, gender))
        conn.commit()
        # Return the ID of the newly inserted customer
        return cursor.lastrowid
def process_frame_thread():
    global CUSTOMERS_IN_FRAME, LAST_SEEN_TIMESTAMP, KNOWN_ENCODINGS, KNOWN_IDS, CUSTOMER_INFO
    
    while True:
        if frame_queue.empty():
            continue
        frame = frame_queue.get()
        if frame is None:
            break
            
        # Update cache periodically
        update_customer_cache()
        
        current_frame_customers = set()
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_locations = [(int(top*2), int(right*2), int(bottom*2), int(left*2)) 
                         for top, right, bottom, left in face_locations]
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(KNOWN_ENCODINGS, face_encoding, tolerance=0.5)
            
            if True in matches:
                customer_id = KNOWN_IDS[matches.index(True)]
                current_frame_customers.add(customer_id)
                
                if customer_id not in CUSTOMERS_IN_FRAME or not CUSTOMERS_IN_FRAME[customer_id]:
                    CUSTOMERS_IN_FRAME[customer_id] = True
                    LAST_SEEN_TIMESTAMP[customer_id] = time.time()
                    update_customer(customer_id, "Camera 1")
                    print(f"New appearance of customer {customer_id} - incrementing visit count")
                
                age = CUSTOMER_INFO[customer_id]['age']
                gender = CUSTOMER_INFO[customer_id]['gender']
                label = f"VIP-{customer_id} ({gender}, {age})"
            else:
                y1, x2, y2, x1 = face_location
                face = frame[y1:y2, x1:x2]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746))
                
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]
                
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age_confidence = age_preds[0]
                age_idx = age_confidence.argmax()
                age = age_list[age_idx]
                confidence = age_confidence[age_idx]
                
                if confidence > 0.6:  # Only accept predictions with >60% confidence
                    age = age_list[age_idx]
                else:
                    age = "Unknown"
                new_id = add_new_customer(face_encoding, "Camera 1", age, gender)
                
                # Update caches immediately after adding new customer
                KNOWN_ENCODINGS.append(face_encoding)
                KNOWN_IDS.append(new_id)
                CUSTOMER_INFO[new_id] = {'age': age, 'gender': gender}
                
                current_frame_customers.add(new_id)
                CUSTOMERS_IN_FRAME[new_id] = True
                LAST_SEEN_TIMESTAMP[new_id] = time.time()
                print(f"New customer {new_id} added to database")
                label = f"New Customer ({gender}, {age})"
            
            y1, x2, y2, x1 = face_location
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Only check for disappearances for customers we're actively tracking
        current_time = time.time()
        for customer_id in list(CUSTOMERS_IN_FRAME.keys()):
            if CUSTOMERS_IN_FRAME[customer_id] and customer_id not in current_frame_customers:
                last_seen = LAST_SEEN_TIMESTAMP.get(customer_id, 0)
                if current_time - last_seen > ABSENCE_THRESHOLD:
                    CUSTOMERS_IN_FRAME[customer_id] = False
                    print(f"Customer {customer_id} has left the frame")
        
        results_queue.put(frame)

def main():
    process_thread = threading.Thread(target=process_frame_thread)
    process_thread.daemon = True
    process_thread.start()
    
    # Initialize database tables
    with sqlite3.connect('customer_data.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS customers (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        face_encoding TEXT,
                        visit_count INTEGER,
                        last_seen TEXT,
                        last_location TEXT,
                        age TEXT,
                        gender TEXT
                    )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS customer_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        customer_id INTEGER,
                        timestamp TEXT,
                        event TEXT,
                        FOREIGN KEY(customer_id) REFERENCES customers(id)
                    )''')
        conn.commit()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        # Process every 3rd frame only
        if frame_count % 3 == 0:
            if not frame_queue.full():
                frame_queue.put(frame)

        # Display processed frame if available
        if not results_queue.empty():
            processed_frame = results_queue.get()
            cv2.imshow('frame', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            frame_queue.put(None)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
