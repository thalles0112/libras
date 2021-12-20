import cv2 as cv
from tensorflow.keras.models import load_model
import numpy as np

labels = ['oi', 'boa', 'tarde']

MODE = 'mod' # normal, mod

opa = '/home/thalles/Documents/PROJETO INTEGRADOR/DATABASE/oi/'
vid_name = ''

model = load_model('modelconv2d-2') # unnoficial
#cap = cv.VideoCapture(opa+vid_name)
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame_ = cv.resize(frame, (300,300))
    gray = cv.cvtColor(frame_, cv.COLOR_BGR2GRAY)
    gray= cv.GaussianBlur(gray, (3, 3),0)
    gray = cv.Canny(gray, 110, 135)
    sobelx = cv.Sobel(gray, cv.CV_64F, 1,0)
    sobely = cv.Sobel(gray, cv.CV_64F, 0,1)
    combined_ = cv.bitwise_or(sobelx, sobely)
    
    combined_p = np.array(combined_, 'float32')
    combined_p = np.expand_dims(combined_p, 0)
    combined_p= np.expand_dims(combined_p, -1)

    pred = np.argmax(model.predict(combined_p))
    print(pred)
    if MODE == 'mod':
        cv.putText(combined_, labels[pred], (30,50), cv.FONT_HERSHEY_COMPLEX, 3, (255,0,0), 3)  #(lal[pred])
        cv.imshow('smile!', combined_)
    elif MODE == 'normal':
        cv.putText(frame, labels[pred], (30,50), cv.FONT_HERSHEY_COMPLEX, 3, (255,0,0), 3)  #(lal[pred])
        cv.imshow('smile!', frame)

    
    
    cv.waitKey(1)

cap.release()
cv.destroyAllWindows()
