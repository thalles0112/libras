import cv2 as cv
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from tensorflow.keras.models import load_model
from kivy.properties import StringProperty

model = load_model('modelconv2d-3')
cap = cv.VideoCapture(0)

predFinal = ''

class OpenCam(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def pred(self, dt):
        labels = ['oi', 'boa', 'tarde']
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

        buf1 = cv.flip(frame, 0)   #inverte para não ficar de cabeça para baixo
        buf = buf1.tostring() # 
        
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            
        self.img1.texture = texture1
        self.predFinal = labels[pred]
        
    def build(self):
        self.img1 = Image(source='logoCL.jpg')
       
        label2= Label(text=StringProperty(self.predFinal))
        layout = BoxLayout(orientation='vertical')  
        
        layout.add_widget(self.img1)
        layout.add_widget(label2)
         
        self.capture = cv.VideoCapture(0)
        ret, frame = self.capture.read() 
        Clock.schedule_interval(self.pred, 1.0/30.0)
        
        return layout
   
if __name__ == '__main__':
    OpenCam().run()