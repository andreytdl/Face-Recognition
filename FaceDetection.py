import cv2
import numpy as np

# Default entry for the notebook webcam is 0
cap = cv2.VideoCapture(0)

#cap = cv2.VideoCapture('avengers.mp4')

#HaarCascade Libraries
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
	
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# Define the fps to be equal to 10. Also frame size is passed.
original_out = cv2.VideoWriter('original.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 21, (frame_width,frame_height))
recognition_output = cv2.VideoWriter('recognized.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 21, (frame_width,frame_height))

#One video is a lot of frames per second, that's why we use while true.
while True:
    
    #ret is the return and will be true or false
    #frame is the image that it is reading
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    # Recording - Write the original frame into the file 'original.avi'
    original_out.write(frame)


    #It makes the BGR frame turns into black and white. 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detects Faces into the gray frame - The numeric values are parameters (See it on the documentation)
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)
	
    #The face starts at X and Y and have size W and H
    for (x,y,w,h) in faces:

        #Setting the square into the frame (face)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        #Taking the square of the gray frame that contains the face
        roi_gray = gray[y:y+h, x:x+w]

        #If you wanna see you can remove the comment below
        ##cv2.imshow("roi_gray", roi_gray) 

        #Taking the square of the colored frame that contains the face
        roi_color = frame[y:y+h, x:x+w]

        #If you wanna see you can remove the comment below
        #cv2.imshow("Face encontrada", roi_color)

        #The eye will only be detected if it is inside the face square
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            #Displaying a green retangle for eyes
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            #Opening a new window to show the eye found
            #cv2.imshow("eye", roi_color[ey:ey+eh, ex:ex+ew])

    

    #showing the camera with all the squares and detections
    cv2.imshow("image", frame)

    # Recording - Write the recognition frame into the file 'recognized.avi'
    recognition_output.write(frame)
    
    #The waitKey receives 1 because it waits for 1 milissecond, displaying the frame on the screen
    #If you wanna see it running 1 FPS is just change 1 for 1000
    key = cv2.waitKey(1) 
    
    if key == 27:
        break

#Releasing and destroying everything
cap.release()
original_out.release()
recognition_output.release()
cv2.destroyAllWindows()