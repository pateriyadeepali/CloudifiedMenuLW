import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from keras.models import model_from_json
import cv2
import copy
from scipy.ndimage import gaussian_filter
import tkinter as tk
import joblib as jb
from keras.preprocessing import image
from PIL import Image
def Camera():
    cap=cv2.VideoCapture(0)
    facemodel=cv2.CascadeClassifier("raw.githubusercontent.com_kipr_opencv_master_data_haarcascades_haarcascade_frontalface_default.xml")
    while True:
        status,photo=cap.read()
        tt=facemodel.detectMultiScale(photo)
        if len(tt)==1:
            x1=tt[0][0]
            y1=tt[0][0]
            x2=tt[0][0]+tt[0][2]
            y2=tt[0][1]+tt[0][3]
            cv2.rectangle(photo,(x1,y1),(x2,y2),[0,255,0],5)
            cv2.imshow("kc",photo)
            if cv2.waitKey(100)==13:
                break
    cv2.destroyAllWindows()
    cap.release()