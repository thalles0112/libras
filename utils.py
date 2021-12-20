import os
import pickle
import cv2 as cv
import sys
import numpy as np

class Utils:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def loadVideo(DIR='pathLike', labels=list):
        paths = []
        training_data = []
        class_num = 0
        for l in labels:
            paths.append(os.path.join(DIR, l))
        
        for path in paths:
            for p in os.listdir(path):
                training_data.append([os.path.join(path,p), class_num])
            class_num += 1

        return training_data
    
    @staticmethod
    def processVideo(training_data=list, n_frames=30, size=(300,300)):
        labels = []
        frames = []
        frame_idx = 0
        
        videos_readed = 0
        n_videos = len(training_data)/100
        sys.stdout.write('\n')
        sys.stdout.write('reading videos...\n')
        sys.stdout.write('[{}]'.format(' '*50))
        sys.stdout.write(('\b'*(50)))

        for t in training_data:
            
            cap = cv.VideoCapture(t[0])
            frame_idx = 0
            while True:
                rec, frame = cap.read()
                
                if rec:
                    frame = cv.resize(frame, size)
                    #frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                   
                    
                    if frame_idx < n_frames:
                        frames.append(frame)
                        
                        labels.append(t[1])
                    
                    elif frame_idx == n_frames:
                        break
                    frame_idx += 1
                else:
                    break

            videos_readed += 1
            
            
            percentage = int(videos_readed / n_videos)
            sys.stdout.write('\b'*((percentage//2)))
            sys.stdout.write('█'*((percentage//2)))
        

            

        print(f'\n {len(frames)} frames readed')
        
        return frames, labels
    
    @staticmethod
    def saveTrainingDatav1(outputFile, object):
        '''
        Desgined to save small datasets. Don't use this
        if you have a large dataset, it will load up memory
        '''
        
        try:
            a = open(outputFile+'.pickle', 'wb')
        except:
            
            a = open(outputFile+'.pickle', 'wb')
            pickle.dump(object, a)
            
        else:
            pickle.dump(object, a)
            a.close()
        a.close()

    @staticmethod
    def loadTrainingData(filename='filename'):
        a = open(filename+'.pickle', 'rb')
        a = pickle.load(a)
        return a


    @staticmethod
    def processVideoMOD(training_data=list, n_frames=30, size=(300,300)):
        labels = []
        frames = []
        
        videos_readed = 0
        n_videos = len(training_data)/100
        sys.stdout.write('\n')
        sys.stdout.write('reading videos...\n')
        sys.stdout.write('[{}]'.format(' '*48))
        sys.stdout.write(('\b'*(49)))

        for t in training_data:
            
            cap = cv.VideoCapture(t[0])
            frame_idx = 0
            while True:
                rec, frame = cap.read()
                
                if rec:
                    frame = cv.resize(frame, size)
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    frame = cv.Canny(frame, 100, 125)
                    framex = cv.Sobel(frame, cv.CV_64F, 1,0)
                    framey = cv.Sobel(frame, cv.CV_64F, 0,1)
                    combined = cv.bitwise_or(framex, framey)
                    combined = np.expand_dims(combined, -1)
                    
                    
                    if frame_idx < n_frames:
                        frames.append(combined)
                        
                        labels.append(t[1])
                    
                    elif frame_idx == n_frames:
                        break
                    frame_idx += 1
                else:
                    break
                    
            videos_readed += 1
                
            percentage = int(videos_readed / n_videos)
            sys.stdout.write('\b'*(int(percentage/2)))
            sys.stdout.write('█'*(int(percentage/2)))
                
        print(f'\n {len(frames)} frames readed')
        return frames, labels


if __name__ == 'main':
    Utils()