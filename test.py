################################################
### This file used to test the trained model ###
################################################


from keras.models import model_from_json
import numpy as np
import os
import cv2 



# Load the trained model 
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Model Loaded")

######################################################
###     Read Images (uncomment the below lines     ###
### And the comment the lines in the Video section)###
######################################################


#img = cv2.imread('./Capture.PNG')
#img=cv2.resize(img,(100,100))
#img = np.expand_dims(img, axis=0)


################################################
###             Read Video                   ###
################################################


# Pass in the file name, normal.mp4 for normal state
# and fire.mp4 for the fire state
cap = cv2.VideoCapture('normal.mp4')  


font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (300,20)
fontScale              = 1
fontColor              = (0,0,255)
lineType               = 2
 
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

# Here we run predictions on each frame of the video

while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    exp_frame=cv2.resize(frame,(100,100))
    exp_frame = np.expand_dims(exp_frame, axis=0)
    if loaded_model.predict(exp_frame)[0][1] == 1.0:
        cv2.putText(frame,'', bottomLeftCornerOfText, font, fontScale,(0,0,0),lineType)
    elif loaded_model.predict(exp_frame)[0][0] == 1.0:
        cv2.putText(frame,'Fire', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
    cv2.imshow('Frame',frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  else: 
    break
 
cap.release()
cv2.destroyAllWindows()
