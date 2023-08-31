import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from keras.models import model_from_json
from cvzone.HandTrackingModule import HandDetector
import pyautogui 
import cv2
import copy
from scipy.ndimage import gaussian_filter
import tkinter as tk
import joblib as jb
from keras.preprocessing import image
from PIL import Image
from FaceRecognition import Camera
from LinearReggression import LinearRegg
from Logg import LogRegg
from blurrface import BlurrtheFace
from Emotion import EmotionDetection
from HandDetection import HandDetect

def predict_class(image_path, model):
    img = Image.open(image_path)
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return 'Dog'
    else:
        return 'Cat'


def DetectDistance():
    REFERENCE_OBJECT_WIDTH = 0.15  # 15 centimeters
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    focal_length = 1000.0
    principal_point = (640, 480)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_width_pixels = w
            distance = (REFERENCE_OBJECT_WIDTH * focal_length) / face_width_pixels

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Distance: {distance:.2f} meters", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

def Volume():
    detector = HandDetector(detectionCon=0.8)
    min_volume = 0
    max_volume = 100
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        hands, frame = detector.findHands(frame)
        if hands:
            for hand in hands:
                landmarks = hand["lmList"]
                bbox = hand["bbox"]
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                thumb_index_distance = np.linalg.norm(np.subtract(thumb_tip, index_tip))
                volume = np.interp(thumb_index_distance, [20, 200], [min_volume, max_volume])
                volume = int(max(min(volume, max_volume), min_volume))
                pyautogui.press('volumedown') if volume < 50 else pyautogui.press('volumeup')
                cv2.putText(frame, f"Volume: {volume}%", (bbox[0], bbox[1] - 20),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.imshow("Volume Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def draw_ashoka_chakra(ax, center_x, center_y, radius):
    # Draw the central blue circle
    circle = plt.Circle((center_x, center_y), radius, color='blue', fill=True)
    ax.add_patch(circle)

    # Draw 24 spokes in the Chakra
    for i in range(24):
        angle = i * 15
        start_x = center_x + 0.45 * radius * np.cos(np.deg2rad(angle))
        start_y = center_y + 0.45 * radius * np.sin(np.deg2rad(angle))
        end_x = center_x + radius * np.cos(np.deg2rad(angle))
        end_y = center_y + radius * np.sin(np.deg2rad(angle))
        ax.plot([start_x, end_x], [start_y, end_y], color='blue', linewidth=1)

    # Draw the small blue circles in the Chakra
    small_circle_radius = radius / 7
    for i in range(4):
        for j in range(6):
            angle = (j * 15) + (i % 2) * 7.5
            x = center_x + 0.7 * radius * np.cos(np.deg2rad(angle))
            y = center_y + 0.7 * radius * np.sin(np.deg2rad(angle))
            small_circle = plt.Circle((x, y), small_circle_radius, color='blue', fill=True)
            ax.add_patch(small_circle)

def draw_indian_flag():
    width, height =805,700
    myimage = np.ones((height, width, 3), dtype=np.uint8) * 255
    myimage[1:225] = [51,153,253]         #for row 
    myimage[225:470] = [255,255,255]
    myimage[470:800]=[8,136,19]
    center=(400,350)
    radius=120
    color=[128,0,0]
    thickness=8
    cv2.circle(myimage, center, radius, color, thickness)
    # Calculate the dimensions of the flag
    stripe_height = height //2
    ashoka_chakra_radius = stripe_height //3
    #Draw the Ashoka Chakra (Navy Blue Circle)
    center_x = width //2
    center_y = stripe_height 
    center = (center_x, center_y)
    cv2.circle(myimage, center, ashoka_chakra_radius,[255,255,255],-1)
    num_lines = 24
    angle = 0
    angle_increment = 360 // num_lines
    for _ in range(num_lines):
        end_x = int(center_x + ashoka_chakra_radius * np.cos(np.deg2rad(angle)))
        end_y = int(center_y + ashoka_chakra_radius * np.sin(np.deg2rad(angle)))
        cv2.line(myimage, center, (end_x, end_y), [128,0,0], thickness=3)
        angle += angle_increment
    cv2.imshow("myimage",myimage)
    cv2.waitKey()
    cv2.destroyAllWindows()    

def DogOrCat():
    loaded_classifier = jb.load("DogvsCat.model")
    a=input("Enter name of the image ")
    new_image_path = a+r'.jpg'
    predicted_class = predict_class(new_image_path, loaded_classifier)
    print(f"The predicted class is: {predicted_class}")

def hello(x):
	#only for referece
	print("")

def HarryPoterCloak():
    cap = cv2.VideoCapture(0)
    bars = cv2.namedWindow("bars")

    cv2.createTrackbar("upper_hue","bars",110,180,hello)
    cv2.createTrackbar("upper_saturation","bars",255, 255, hello)
    cv2.createTrackbar("upper_value","bars",255, 255, hello)
    cv2.createTrackbar("lower_hue","bars",68,180, hello)
    cv2.createTrackbar("lower_saturation","bars",55, 255, hello)
    cv2.createTrackbar("lower_value","bars",54, 255, hello)

    #Capturing the initial frame for creation of background
    while(True):
        cv2.waitKey(1000)
        ret,init_frame = cap.read()
        #check if the frame is returned then brake
        if(ret):
            break

    # Start capturing the frames for actual magic!!
    while(True):
        ret,frame = cap.read()
        inspect = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #getting the HSV values for masking the cloak
        upper_hue = cv2.getTrackbarPos("upper_hue", "bars")
        upper_saturation = cv2.getTrackbarPos("upper_saturation", "bars")
        upper_value = cv2.getTrackbarPos("upper_value", "bars")
        lower_value = cv2.getTrackbarPos("lower_value","bars")
        lower_hue = cv2.getTrackbarPos("lower_hue","bars")
        lower_saturation = cv2.getTrackbarPos("lower_saturation","bars")

        #Kernel to be used for dilation
        kernel = np.ones((3,3),np.uint8)

        upper_hsv = np.array([upper_hue,upper_saturation,upper_value])
        lower_hsv = np.array([lower_hue,lower_saturation,lower_value])

        mask = cv2.inRange(inspect, lower_hsv, upper_hsv)
        mask = cv2.medianBlur(mask,3)
        mask_inv = 255-mask 
        mask = cv2.dilate(mask,kernel,5)
        
        #The mixing of frames in a combination to achieve the required frame
        b = frame[:,:,0]
        g = frame[:,:,1]
        r = frame[:,:,2]
        b = cv2.bitwise_and(mask_inv, b)
        g = cv2.bitwise_and(mask_inv, g)
        r = cv2.bitwise_and(mask_inv, r)
        frame_inv = cv2.merge((b,g,r))

        b = init_frame[:,:,0]
        g = init_frame[:,:,1]
        r = init_frame[:,:,2]
        b = cv2.bitwise_and(b,mask)
        g = cv2.bitwise_and(g,mask)
        r = cv2.bitwise_and(r,mask)
        blanket_area = cv2.merge((b,g,r))

        final = cv2.bitwise_or(frame_inv, blanket_area)

        cv2.imshow("Harry's Cloak",final)

        if(cv2.waitKey(3) == ord('q')):
            break;

    cv2.destroyAllWindows()
    cap.release()

def Crop():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Step 4: Capture Video from Webcam (Optional)
    cap = cv2.VideoCapture(0)
    # Step 5: Process the Video Stream or Load an Image
    while True:
        ret, img = cap.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Step 6: Perform Face Detection
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # Step 7: Draw Rectangles around Detected Faces and Display in a Window
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Crop the face region and display it in a separate window
            face_roi = img[y:y + h, x:x + w]
            cv2.imshow('Detected Face', face_roi)
        cv2.imshow('Face Detection', img)
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def BlurrtheSurr():
    cap = cv2.VideoCapture(0)
    face_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    while True:
        ret, photo = cap.read()

        if not ret:
            break

        faces = face_model.detectMultiScale(photo)

        # Apply Gaussian blur to the background
        blurred_photo = blur_background(photo.copy(), faces)

        # Draw green rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(blurred_photo, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Blurred Background", blurred_photo)

        if cv2.waitKey(1) & 0xFF == 13:
            break

    cv2.destroyAllWindows()
    cap.release()

def blur_background(image, faces):
    # Create a mask to separate the face region from the background
    mask = image.copy()
    mask[:] = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
    inverted_mask = cv2.bitwise_not(mask)

    # Apply Gaussian blur only to the background
    blurred_background = cv2.GaussianBlur(image, (23, 23), 30)
    result = cv2.bitwise_and(image, mask) + cv2.bitwise_and(blurred_background, inverted_mask)
    return result

def BlurrtheFace():
    cap = cv2.VideoCapture(0)
    face_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    while True:
        ret, photo = cap.read()

        if not ret:
            break

        faces = face_model.detectMultiScale(photo)

        # Create a copy of the original photo
        blurred_photo = photo.copy()

        # Apply Gaussian blur to the face regions
        blurred_photo = blur_face(blurred_photo, faces)

        cv2.imshow("Blurred Face", blurred_photo)

        if cv2.waitKey(1) & 0xFF == 13:
            break

    cv2.destroyAllWindows()
    cap.release()

def blur_face(image, faces):
    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_roi, (23, 23), 30)
        image[y:y+h, x:x+w] = blurred_face
    return image


def execute_selected_task():
    selected_task = int(choice_var.get())
    if selected_task == 1:
        LinearRegg()
    elif selected_task == 2:
        LogRegg()
    elif selected_task == 3:
        HarryPoterCloak()
    elif selected_task == 4:
        BlurrtheFace()
    elif selected_task == 5:
        BlurrtheSurr()
    elif selected_task == 6:
        DetectDistance()
    elif selected_task == 7:
        EmotionDetection()
    elif  selected_task==8:
        draw_indian_flag()
    elif selected_task==9:
        DogOrCat()
    elif selected_task==10:
        HandDetect()
    elif selected_task==11:
        Volume()
    elif selected_task==12:
        Crop()
# Create the Tkinter GUI
root = tk.Tk()
root.title("Task Menu")
window_width = 400
window_height = 500
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = int((screen_width - window_width) / 2)
y_coordinate = int((screen_height - window_height) / 2)
root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

# Create a label to display the menu options
label = tk.Label(root, text="Team Tecnoits \nSelect a task:")
label.pack()

# Create a variable to hold the user's choice
choice_var = tk.StringVar()

# Create radio buttons for the user to select a task
tk.Radiobutton(root, text="Run Regression Model", variable=choice_var, value="1").pack(anchor=tk.W)
tk.Radiobutton(root, text="Run Logistic Regression", variable=choice_var, value="2").pack(anchor=tk.W)
tk.Radiobutton(root, text="Harry Poter Cloak", variable=choice_var, value="3").pack(anchor=tk.W)
tk.Radiobutton(root, text="Blurring the Face", variable=choice_var, value="4").pack(anchor=tk.W)
tk.Radiobutton(root, text="Blurring the Surrounding", variable=choice_var, value="5").pack(anchor=tk.W)
tk.Radiobutton(root, text="Detect Distance", variable=choice_var, value="6").pack(anchor=tk.W)
tk.Radiobutton(root, text="Emotion Detection", variable=choice_var, value="7").pack(anchor=tk.W)
tk.Radiobutton(root, text="Indian Flag", variable=choice_var, value="8").pack(anchor=tk.W)
tk.Radiobutton(root, text="Dog or Cat", variable=choice_var, value="9").pack(anchor=tk.W)
tk.Radiobutton(root, text="Hand Detection ", variable=choice_var, value="10").pack(anchor=tk.W)
tk.Radiobutton(root, text="Increase Volume ", variable=choice_var, value="11").pack(anchor=tk.W)
tk.Radiobutton(root, text="Cropped Image", variable=choice_var, value="12").pack(anchor=tk.W)


# Create a button to execute the selected task
execute_button = tk.Button(root, text="Execute Task", command=execute_selected_task)
execute_button.pack()

# Start the Tkinter main loop
root.mainloop()