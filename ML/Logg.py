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
def LogRegg():
    dataset = pd.read_csv(input("Enter the name of your dataset ")+".csv")
    x= dataset.iloc[ : , 0:8]
    y= dataset['Outcome'] 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print("Accuracy:", accuracy)
    import joblib as jb
    jb.dump(model,"Logistic.model")