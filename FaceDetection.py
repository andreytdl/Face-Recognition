import cv2
import numpy as np

# Por Default a webcam fica no 0
cap = cv2.VideoCapture(0)

#Bibliotecas HaarCascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')



#Um vídeo são vários frames passando por segundo por isso while true
while True:
    
    #ret é o retorno e irá vir como true ou false
    #frame é a propria imagem em si
    ret, frame = cap.read()

    #Deixa o frame que é BGR em preto e branco 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detecta Faces dentro do gray - Os valores numéricos são padrões
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	
    #A face começa no X e no Y e tem o tamanho W e H
    for (x,y,w,h) in faces:
        
        #Colocando o quadrado dentro de frame
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        roi_gray = gray[y:y+h, x:x+w]
        ##cv2.imshow("roi_gray", roi_gray) 

        roi_color = frame[y:y+h, x:x+w]
        #cv2.imshow("Face encontrada", roi_color)

        #O olho será detectado apenas dentro da face
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.imshow("olho", roi_color[ey:ey+eh, ex:ex+ew])

    


    #para preto e branco trocamos o "frame" por gray
    cv2.imshow("frame", frame)

    
    #No waitKey esperamos 1 milesimo de segundo, se colocarmos 1000 os frames serão 1 frame por segundo/
    #Nada mais é do que os frames por segundo
    key = cv2.waitKey(1) 
    
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()