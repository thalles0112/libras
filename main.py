from kivy.uix.label import Label
import cv2 as cv
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from keras.models import load_model
from kivy.lang import Builder

model = load_model('modelconv2d-3')


class OpenCam(App):
    img1= Image()
    mytext = Label(text='aa')
    camIndex = 0
    def build(self):
        Clock.schedule_interval(self.pred, 1.0/30.0)
        
        box = BoxLayout()
        box.orientation = 'vertical'
        label = self.mytext
        img = self.img1
        box.add_widget(img)
        box.add_widget(label)
        return box
 
    def pred(self, *args):
        cap = cv.VideoCapture(self.camIndex)
        labels = ['oi', 'boa', 'tarde']
        ret, frame = cap.read()
        frame_ = cv.resize(frame, (300,300))
        frame_ = np.array(frame_, 'float32')
        frame_ = np.expand_dims(frame_, 0)
        frame_ = np.expand_dims(frame_, -1)

        pred = np.argmax(model.predict(frame_))

        buf1 = cv.flip(frame, 0)   #inverte para não ficar de cabeça para baixo
        buf = buf1.tostring() # 
        
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            
        self.img1.texture = texture1
        self.mytext.text = (labels[pred]).__str__()
    def changeCam(self):
        if self.camIndex == 0:
            self.camIndex = 1
        elif self.camIndex == 1:
            self.camIndex = 0
        


if __name__ == '__main__':
    OpenCam().run()