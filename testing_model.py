import tensorflow as tf
import tflearn
from tflearn.layers.recurrent import lstm
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected, time_distributed, flatten
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import cv2
import imutils

bg = None

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    (cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def main():
    aWeight = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0
    start_recording = False

    while(True):
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width = 700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if num_frames < 5:
            run_avg(gray, aWeight)
        else:
            hand = segment(gray)
            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                if start_recording:
                    cv2.imwrite('Temp.png', thresholded)
                    resizeImage('Temp.png')
                    predictedClass, confidence = getPredictedClass()
                    showStatistics(predictedClass, confidence)
                cv2.imshow("Thesholded", thresholded)

        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        num_frames += 1
        cv2.imshow("Video Feed", clone)
        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break

        start_recording = True

def getPredictedClass():
    # Predict
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([gray_image.reshape(89, 100, 1)])
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2]))

def showStatistics(predictedClass, confidence):

    textImage = np.zeros((300,512,3), np.uint8)
    className = ""

    if predictedClass == 0:
        className = "Gunting"
    elif predictedClass == 1:
        className = "Batu"
    elif predictedClass == 2:
        className = "Kertas"

    cv2.putText(textImage,"Pedicted Class : " + className,
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    # cv2.putText(textImage,"Confidence : " + str(confidence * 100) + '%',
    #             (30, 100),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             1,
    #             (255, 255, 255),
    #             2)

    cv2.imshow("Statistics", textImage)
    print(className+" "+str(confidence))

tf.reset_default_graph()
cnnlstmmodel=input_data(shape=[None,89,100,1],name='input')

cnnlstmmodel=conv_2d(cnnlstmmodel,32,2,activation='relu')
cnnlstmmodel=max_pool_2d(cnnlstmmodel,2)

cnnlstmmodel=conv_2d(cnnlstmmodel,64,2,activation='relu')
cnnlstmmodel=max_pool_2d(cnnlstmmodel,2)

cnnlstmmodel=conv_2d(cnnlstmmodel,128,2,activation='relu')
cnnlstmmodel=max_pool_2d(cnnlstmmodel,2)

cnnlstmmodel=conv_2d(cnnlstmmodel,256,2,activation='relu')
cnnlstmmodel=max_pool_2d(cnnlstmmodel,2)

cnnlstmmodel=conv_2d(cnnlstmmodel,512,2,activation='relu')
cnnlstmmodel=max_pool_2d(cnnlstmmodel,2)

cnnlstmmodel=conv_2d(cnnlstmmodel,256,2,activation='relu')
cnnlstmmodel=max_pool_2d(cnnlstmmodel,2)

cnnlstmmodel=conv_2d(cnnlstmmodel,128,2,activation='relu')
cnnlstmmodel=max_pool_2d(cnnlstmmodel,2)

cnnlstmmodel=conv_2d(cnnlstmmodel,64,2,activation='relu')
cnnlstmmodel=max_pool_2d(cnnlstmmodel,2)

cnnlstmmodel=conv_2d(cnnlstmmodel,32,2,activation='relu')
cnnlstmmodel=max_pool_2d(cnnlstmmodel,2)

cnnlstmmodel = time_distributed(cnnlstmmodel, flatten, args=['flat'])
cnnlstmmodel = lstm(cnnlstmmodel, 1024)

cnnlstmmodel=fully_connected(cnnlstmmodel,500,activation='relu')
cnnlstmmodel=dropout(cnnlstmmodel,0.7)

cnnlstmmodel=fully_connected(cnnlstmmodel,3,activation='softmax')

cnnlstmmodel=regression(cnnlstmmodel,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')
model=tflearn.DNN(cnnlstmmodel,tensorboard_verbose=0)

# Load Saved Model
model.load("modelmaker/cnnlstm.tfl")

main()
