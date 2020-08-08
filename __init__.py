
import os
import cv2 as cv
from flask import Flask,request,render_template

from os.path import join as pjoin
import keras
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib
import shutil

datapath = '/data/visualsearch'
