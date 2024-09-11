import os
from ultralytics import YOLO
import cv2
import pygame


os.chdir(r"F:\YOLO Projects\Distracted_drivers")

pygame.init()
pygame.mixer.music.load("alarm.wav")

model = YOLO("BestDriver.pt")

cap = cv2.VideoCapture("video1.mp4")

Classes = model.names
# print(Classes)

def draw_text_with_background(frame, text, position, font, scale, text_color, background_color, border_color, thickness=2, padding=5):
    """Draw text with background and border on the frame."""
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    # Background rectangle
    cv2.rectangle(frame, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  background_color, 
                  cv2.FILLED)
    # Border rectangle
    cv2.rectangle(frame, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  border_color, 
                  thickness)
    # Text
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, lineType=cv2.LINE_AA)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("The frame have been finished")
        break
    else:
        frame = cv2.resize(frame, (1100, 700))
        results = model.predict(frame, conf=0.65)
        for result in results:
            Boxes = result.boxes.xyxy
            Labels = result.boxes.cls
            Confs = result.boxes.conf
            for box, label, conf in zip(Boxes, Labels, Confs):
                x, y, w, h = map(int, box)
                label = int(label)
                conf = float(conf)
                if label == 0:
                    cv2.rectangle(frame, (x, y), (w, h), [0, 255, 0], 2)
                    draw_text_with_background(frame, 
                                      f"{Classes[label].capitalize()}, Conf{(conf*100):0.2f}%", 
                                      (x, y - 10), 
                                      cv2.FONT_HERSHEY_COMPLEX, 
                                      0.6, 
                                      (255, 255, 255),  # White text
                                      (0, 0, 0),  # Black background
                                      (0, 255, 0))  
                    
                else:
                    cv2.rectangle(frame, (x, y), (w, h), [0, 0, 255], 2)
                    draw_text_with_background(frame, 
                                      f"{Classes[label].capitalize()}, Conf{(conf*100):0.2f}%", 
                                      (x, y - 10), 
                                      cv2.FONT_HERSHEY_COMPLEX, 
                                      0.6, 
                                      (255, 255, 255),  # White text
                                      (0, 0, 0),  # Black background
                                      (0, 0, 255))  #  border
                    if not pygame.mixer.music.get_busy():
                            pygame.mixer.music.play()
                    
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 27:
            break
        
cap.release()
cv2.destroyAllWindows()