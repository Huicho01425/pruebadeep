# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 18:16:54 2021

@author: Jonathan Sanchez
"""
import cv2 as cv
import imutils

cascPath="haarcascade_frontalface_default.xml"
faceCascade=cv.CascadeClassifier(cascPath)


videoCapture=cv.VideoCapture(0)
filtro=cv.imread("calabaza.png",cv.IMREAD_UNCHANGED)
salida = cv.VideoWriter('videoPract7.avi',cv.VideoWriter_fourcc(*'XVID'),20.0,(720,480))
if(not videoCapture.isOpened()):
    print("No se pudo abrir la camara")
else:
    while True:
        ret,frame=videoCapture.read()
        frame=cv.flip(frame,1, (720,480))
        imagenGrises=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(imagenGrises,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
        
        for(x,y,w,h) in faces:  
            #cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),10)
            
            #Redimensionar la imagen de entrada
            resized_image = imutils.resize(filtro, width=w)
            filas_image = resized_image.shape[0]
            col_image = w
            
            dif = 0
            
            porcion_alto = filas_image // 4
            
            if y - filas_image + porcion_alto >= 0:
                n_frame = frame[y - filas_image + porcion_alto: y + porcion_alto, x: x + w]
                
            else:
                dif = abs(y - filas_image + porcion_alto)
                n_frame = frame[0: y + porcion_alto, x: x + w]
                
            mask = resized_image[:, :, 3]
            mask_inv = cv.bitwise_not(mask)
                
            bg_black = cv.bitwise_and(resized_image,resized_image, mask=mask)
            bg_black = bg_black[dif:, :, 0:3]
            bg_frame = cv.bitwise_and(n_frame,n_frame, mask=mask_inv[dif:, :])
            
            #sumamos las dos imagenes
            result = cv.add(bg_black, bg_frame)
            
            if y - filas_image + porcion_alto >= 0:
                frame[y - filas_image + porcion_alto: y + porcion_alto, x: x + w] = result
                cv.imshow('Practica 7',frame)
                salida.write(frame)
                
            else:
                frame[0: y + porcion_alto, x: x + w] = result
                cv.imshow('Practica 7',frame)
                salida.write(frame)
        
        if(cv.waitKey(1) & 0xFF==ord('s')):
            break
        
    videoCapture.release()
    salida.release()
    cv.destroyAllWindows()
        