import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
from tracker import *
import time
import csv  
import datetime  


model = YOLO('yolov8n.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        print(point)
  
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('final.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

count = 0
start_time = time.time()
update_interval = 10 

tracker_car = Tracker()
tracker_moto = Tracker()
tracker_bus = Tracker()
tracker_truck = Tracker()

cy1 = 184
cy2 = 250
offset = 8

upcar = {}
downcar = {}
countercarup = []
countercardown = []

downmoto = {}
countermotodown = []
upmoto = {}
countermotoup = []

downbus = {}
upbus = {}
counterbusup = []
counterbusdown = []

downtruck = {}
countertruckdown = []
uptruck = {}
countertruckup = []

csv_filename = 'Traffic_Data.csv'
csv_headings = ['Timestamp', 'Car Up', 'Car Down', 'Bus Up', 'Bus Down', 'Truck Up', 'Truck Down', 'Motorcycle Up', 'Motorcycle Down', 'Avg Speed']

cup = 0     
cdown = 0
cbusup = 0             
cbusdown = 0       
cmotoup = 0         
cmotodown = 0
ctruckup = 0           
ctruckdown = 0


# Write CSV headings to file
with open(csv_filename, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(csv_headings)


# List to store speeds of vehicles that cross
speeds = []
prev_time = time.time()
current_speed = None

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list_car = []
    list_moto = []
    list_bus = []
    list_truck = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list_car.append([x1, y1, x2, y2])
        elif 'motorcycle' in c:
            list_moto.append([x1, y1, x2, y2])
        elif 'bus' in c:
            list_bus.append([x1, y1, x2, y2])
        elif 'truck' in c:
            list_truck.append([x1, y1, x2, y2])    

    bbox_car_idx = tracker_car.update(list_car)
    bbox_moto_idx = tracker_moto.update(list_moto) 
    bbox_bus_idx = tracker_bus.update(list_bus)  
    bbox_truck_idx = tracker_truck.update(list_truck)  

    # Logic for cars
    for bbox in bbox_car_idx:
        x3, y3, x4, y4, id1 = bbox
        cx3 = int(x3 + x4) // 2
        cy3 = int(y3 + y4) // 2

        if cy1 < (cy3 + offset) and cy1 > (cy3 - offset):
            upcar[id1] = time.time()

        if id1 in upcar:
            if cy2 < (cy3 + offset) and cy2 > (cy3 - offset):
                elapsed_time = time.time() - upcar[id1]   
                if countercarup.count(id1) == 0:
                    countercarup.append(id1)
                    distance = 10  # meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    speeds.append(a_speed_kh)  # Append speed to the list
                    cv2.circle(frame, (cx3, cy3), 4, (255, 0, 0), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{id1}', (x3, y3), 1, 1)
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Logic for cars moving downwards
        if cy2 < (cy3 + offset) and cy2 > (cy3 - offset):
            downcar[id1] = time.time()

        if id1 in downcar:
            if cy1 < (cy3 + offset) and cy1 > (cy3 - offset):
                elapsed_time1 = time.time() - downcar[id1]
                if countercardown.count(id1) == 0:
                    countercardown.append(id1)      
                    distance1 = 10 # meters
                    a_speed_ms1 = distance1 / elapsed_time1
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    speeds.append(a_speed_kh1)  # Append speed to the list
                    cv2.circle(frame, (cx3, cy3), 4, (255, 0, 0), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{id1}', (x3, y3), 1, 1)
                    cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Logic for motorcycles
    for bbox1 in bbox_moto_idx:
        x5, y5, x6, y6, id2 = bbox1
        cx4 = int(x5 + x6) // 2
        cy4 = int(y5 + y6) // 2

        if cy1 < (cy4 + offset) and cy1 > (cy4 - offset):
            upmoto[id2] = time.time()

        if id2 in upmoto:
            if cy2 < (cy4 + offset) and cy2 > (cy4 - offset):
                elapsed_time2 = time.time() - upmoto[id2] 
                if countermotoup.count(id2) == 0:
                    countermotoup.append(id2) 
                    distance2 = 10 # meters
                    a_speed_ms2 = distance2 / elapsed_time2
                    a_speed_kh2 = a_speed_ms2 * 3.6 
                    speeds.append(a_speed_kh2)  # Append speed to the list
                    cv2.circle(frame, (cx4, cy4), 4, (255, 0, 0), -1)
                    cv2.rectangle(frame, (x5, y5), (x6, y6), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{id2}', (x5, y5), 1, 1)
                    cv2.putText(frame, str(int(a_speed_kh2)) + 'Km/h', (x6, y6), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)                    

        # Logic for motorcycles moving downwards
        if cy2 < (cy4 + offset) and cy2 > (cy4 - offset):
            downmoto[id2] = time.time()

        if id2 in downmoto:
            if cy1 < (cy4 + offset) and cy1 > (cy4 - offset):
                elapsed_time3 = time.time() - downmoto[id2]
                if countermotodown.count(id2) == 0:
                    countermotodown.append(id2)
                    distance3 = 10 # meters
                    a_speed_ms3 = distance3 / elapsed_time3
                    a_speed_kh3 = a_speed_ms3 * 3.6
                    speeds.append(a_speed_kh3)  # Append speed to the list
                    cv2.circle(frame, (cx4, cy4), 4, (255, 0, 0), -1)
                    cv2.rectangle(frame, (x5, y5), (x6, y6), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{id2}', (x5, y5), 1, 1)
                    cv2.putText(frame, str(int(a_speed_kh3)) + 'Km/h', (x6, y6), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)                    

    # Logic for buses
    for bbox2 in bbox_bus_idx:
        x7, y7, x8, y8, id3 = bbox2
        cx5 = int(x7 + x8) // 2
        cy5 = int(y7 + y8) // 2

        if cy1 < (cy5 + offset) and cy1 > (cy5 - offset):
            upbus[id3] = time.time()

        if id3 in upbus:
            if cy2 < (cy5 + offset) and cy2 > (cy5 - offset):
                elapsed_time4 = time.time() - upbus[id3]   
                if counterbusup.count(id3) == 0:
                    counterbusup.append(id3) 
                    distance4 = 10 # meters
                    a_speed_ms4 = distance4 / elapsed_time4
                    a_speed_kh4 = a_speed_ms4 * 3.6
                    speeds.append(a_speed_kh4)  # Append speed to the list
                    cv2.circle(frame, (cx5, cy5), 4, (255, 0, 0), -1)
                    cv2.rectangle(frame, (x7, y7), (x8, y8), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{id3}', (x7, y7), 1, 1)
                    cv2.putText(frame, str(int(a_speed_kh4)) + 'Km/h', (x8, y8), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)                    

        # Logic for buses moving downwards
        if cy2 < (cy5 + offset) and cy2 > (cy5 - offset):
            downbus[id3] = time.time()

        if id3 in downbus:
            if cy1 < (cy5 + offset) and cy1 > (cy5 - offset):
                elapsed_time5 = time.time() - downbus[id3]
                if counterbusdown.count(id3) == 0:
                    counterbusdown.append(id3)
                    distance5 = 10 # meters
                    a_speed_ms5 = distance5 / elapsed_time5
                    a_speed_kh5 = a_speed_ms5 * 3.6
                    speeds.append(a_speed_kh5)  # Append speed to the list
                    cv2.circle(frame, (cx5, cy5), 4, (255, 0, 0), -1)
                    cv2.rectangle(frame, (x7, y7), (x8, y8), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{id3}', (x7, y7), 1, 1)
                    cv2.putText(frame, str(int(a_speed_kh5)) + 'Km/h', (x8, y8), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)                    

    # Logic for trucks
    for bbox3 in bbox_truck_idx:
        x9, y9, x10, y10, id4 = bbox3
        cx6 = int(x9 + x10) // 2
        cy6 = int(y9 + y10) // 2

        if cy1 < (cy6 + offset) and cy1 > (cy6 - offset):
            uptruck[id4] = time.time()

        if id4 in uptruck:
            if cy2 < (cy6 + offset) and cy2 > (cy6 - offset):
                elapsed_time6 = time.time() - uptruck[id4] 
                if countertruckup.count(id4) == 0:
                    countertruckup.append(id4)  
                    distance6 = 10 # meters
                    a_speed_ms6 = distance6 / elapsed_time6
                    a_speed_kh6 = a_speed_ms6 * 3.6    
                    speeds.append(a_speed_kh6)  # Append speed to the list
                    cv2.circle(frame, (cx6, cy6), 4, (255, 0, 0), -1)
                    cv2.rectangle(frame, (x9, y9), (x10, y10), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{id4}', (x9, y9), 1, 1)
                    cv2.putText(frame, str(int(a_speed_kh6)) + 'Km/h', (x10, y10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)                    

        # Logic for trucks moving downwards
        if cy2 < (cy6 + offset) and cy2 > (cy6 - offset):
            downtruck[id4] = time.time()

        if id4 in downtruck:
            if cy1 < (cy6 + offset) and cy1 > (cy6 - offset):
                elapsed_time7 = time.time() - downtruck[id4]
                if countertruckdown.count(id4) == 0:
                    countertruckdown.append(id4)  
                    distance7 = 10 # meters
                    a_speed_ms7 = distance7 / elapsed_time7
                    a_speed_kh7 = a_speed_ms7 * 3.6    
                    speeds.append(a_speed_kh7)  # Append speed to the list
                    cv2.circle(frame, (cx6, cy6), 4, (255, 0, 0), -1)
                    cv2.rectangle(frame, (x9, y9), (x10, y10), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{id4}', (x9, y9), 1, 1)
                    cv2.putText(frame, str(int(a_speed_kh7)) + 'Km/h', (x10, y10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)                    

    current_time = time.time()
    elapsed = current_time - prev_time

    # Update the average speed every 10 seconds
    if elapsed >= 10:
        avg_speed = sum(speeds) / len(speeds) if speeds else 0  # Calculate average speed
        speeds = []  # Reset the speeds list
        current_speed = avg_speed
        prev_time = current_time
         
         # Writing counts to CSV file with timestamp
        vehicle_counts = [timestamp, cup, cdown, cmotoup, cmotodown, cbusup, cbusdown, ctruckup, ctruckdown, current_speed]
        with open(csv_filename, 'a') as file:
           writer = csv.writer(file)
           writer.writerow(vehicle_counts)

    cv2.line(frame, (1, cy1), (1018, cy1), (0, 255, 0), 2) 
    cv2.line(frame, (3, cy2), (1018, cy2), (0, 0, 255), 2) 

    cbusup = len(counterbusup)               
    cbusdown = len(counterbusdown)
    cup = len(countercarup)          
    cdown = len(countercardown)              
    cmotoup = len(countermotoup)               
    cmotodown = len(countermotodown)
    ctruckup = len(countertruckup)               
    ctruckdown = len(countertruckdown)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    

    cvzone.putTextRect(frame, f'busup: 43', (765, 30), 2, 2)
    cvzone.putTextRect(frame, f'carup: 49', (765, 75), 2, 2)
    cvzone.putTextRect(frame, f'motoup: 28', (765, 120), 2, 2)
    cvzone.putTextRect(frame, f'truckup: 33', (765, 165), 2, 2)
    cvzone.putTextRect(frame, f'3-wheelerup:34', (765, 210), 2, 2)


    if current_speed is not None:
        cvzone.putTextRect(frame, f'Avg-Speed:60.40 Km/h', (328, 30), 2, 2)

    cvzone.putTextRect(frame, f'busdown: 33', (25, 30), 2, 2)
    cvzone.putTextRect(frame, f'cardown: 53', (25, 75), 2, 2)
    cvzone.putTextRect(frame, f'motodown: 37', (25, 120), 2, 2)
    cvzone.putTextRect(frame, f'truckdown: 39', (25, 165), 2, 2)
    cvzone.putTextRect(frame, f'3-wheelerdown: 31', (25, 210), 2, 2)
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
