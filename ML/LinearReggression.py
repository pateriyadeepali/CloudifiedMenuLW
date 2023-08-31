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
def LinearRegg():
    data=input("Enter the name of the file : ")
    dataset=pd.read_csv(data+".csv")
    last_column = dataset.columns[-1]
    y=dataset[last_column]
    X = dataset.iloc[:, :-1]
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
    model=LinearRegression()
    model.fit(X_train,y_train)
    y_predict=model.predict(X_test)
    jb.dump(model,"Regression_Model.model")