import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

video = cv.VideoCapture("videoMultiObjectTracking4.mp4")
fps = video.get(cv.CAP_PROP_FPS)

# videoWriter = cv.VideoWriter("videoProcessedMultiObjectTrack.avi",cv.VideoWriter_fourcc(*"DIVX"), fps,(1080,1920))
videoWriter = cv.VideoWriter("videoProcessedMultiObjectTrackBlurred.avi",cv.VideoWriter_fourcc(*"DIVX"), fps,(1080,1920))
# videoWriter = cv.VideoWriter("videoMultiObjectTrackingBlurred.avi",cv.VideoWriter_fourcc(*"DIVX"), fps,(1080,1920))


person = cv.imread("arm.png")
c, persW, persH, = person.shape[::-1]
car = cv.imread("car42.png")
c, carW, carH= car.shape[::-1]
car2 = cv.imread("car4N2.png")
c, car2W, car2H= car2.shape[::-1]
bus = cv.imread("bus4.png")
c, busW, busH, = bus.shape[::-1]

cv.namedWindow("Resized_Window", cv.WINDOW_NORMAL)  
fps = video.get(cv.CAP_PROP_FPS)

width = video.get(cv.CAP_PROP_FRAME_WIDTH)
height = video.get(cv.CAP_PROP_FRAME_HEIGHT)
video.set(cv.CAP_PROP_POS_MSEC, 0)

# endtime1 = 10000
# endtime2 = 20000
# endtime3 = 40000

initializedCarTracker = 0
initializedCar2Tracker = 0
initializedPersTracker = 0
initializedBusTracker = 0

cv.resizeWindow("Resized_Window", round(width/3), round(height/3)) 

# ret, frame = video.read()
# if ret == True:
#     personMatch = cv.matchTemplate(frame, person, cv.TM_CCOEFF) 
#     persMinVal, persMaxVal, persMinLoc, persMaxLoc = cv.minMaxLoc(personMatch)
#     cv.rectangle(frame, [persMaxLoc[0],persMaxLoc[1]-persH], [persMaxLoc[0]+persW, persMaxLoc[1]+persH], (0, 255, 0), 3 )
#     boxPers = [persMaxLoc[0], persMaxLoc[1]-persH, 2*persW, 2*persH]
    
    # print(persMaxLoc[0], persMaxLoc[1]-persH, persMaxLoc[0]+persW, persMaxLoc[1]+persH)
    # carMatch = cv.matchTemplate(frame, car, cv.TM_CCOEFF) 
persTracker = cv.TrackerKCF_create()
carTracker = cv.TrackerKCF_create()
car2Tracker = cv.TrackerKCF_create()

busTracker = cv.TrackerKCF_create()
# busTracker = cv.legacy_TrackerMedianFlow()


while(video.isOpened()):
    if video.get(cv.CAP_PROP_POS_MSEC) < 21000:
        ret, frame = video.read()
        blurred= frame
        if ret == True:
            personMatch = cv.matchTemplate(frame, person, cv.TM_SQDIFF)  
            persMinVal, persMaxVal, persMinLoc, persMaxLoc = cv.minMaxLoc(personMatch)

            # Logic to find initial position of the tracked objects
            personMatchClean = np.float32(personMatch < 21375950.0)
            
            persMinVal, persMaxVal, persMinLoc, persMaxLoc = cv.minMaxLoc(personMatchClean)

            carMatch = cv.matchTemplate(frame, car, cv.TM_CCOEFF)
            carMinVal, carMaxVal, carMinLoc, carMaxLoc = cv.minMaxLoc(carMatch)  
            # print(carMaxVal)
            carMatchClean = np.float32(carMatch > 57568040)
            carMinVal, carMaxVal, carMinLoc, carMaxLoc = cv.minMaxLoc(carMatchClean)

            car2Match = cv.matchTemplate(frame, car2, cv.TM_SQDIFF)
            car2MinVal, car2MaxVal, car2MinLoc, car2MaxLoc = cv.minMaxLoc(car2Match)  
            # print(car2MinVal)
            car2MatchClean = np.float32(car2Match < 37697768.0) # 27259920.0
            car2MinVal, car2MaxVal, car2MinLoc, car2MaxLoc = cv.minMaxLoc(car2MatchClean)
            
            busMatch = cv.matchTemplate(frame, bus, cv.TM_CCOEFF)
            busMinVal, busMaxVal, busMinLoc, busMaxLoc = cv.minMaxLoc(busMatch)  
            # print(busMaxVal)
            busMaxValReal = busMaxVal
            busMatchClean = np.float32(busMatch > 320000712.0)
            busMinVal, busMaxVal, busMinLoc, busMaxLoc = cv.minMaxLoc(busMatchClean)
            
            #white car tracking
            if carMaxVal == 1:
                if initializedCarTracker == 0:
                    cv.rectangle(frame, [carMaxLoc[0]-carW//2-20, carMaxLoc[1]], [carMaxLoc[0]+carW*3//2-20, carMaxLoc[1]+2*carH], (255, 0, 0), 3 )
                    boxCar = [carMaxLoc[0]-carW//2-20, carMaxLoc[1], 2*carW, 2*carH]
                    # print(carMaxLoc[0]-carW+10, carMaxLoc[1]+10, 2*carW, 2*carH)
                    carTracker.init(frame, boxCar)
                    initializedCarTracker = 1
            if initializedCarTracker == 1:
                 retCar, boxCar = carTracker.update(frame)
                 if retCar:
                    p1Car = (int(boxCar[0]), int(boxCar[1]))
                    p2Car = (int(boxCar[0] + boxCar[2]), int(boxCar[1] + boxCar[3]))
                    cv.rectangle(frame, p1Car, p2Car, (255, 0, 0), 3, 1)
                    cv.putText(frame, "Car 1", [p1Car[0], p1Car[1]-10], 0 , 1.7, (255, 0, 0), 3)
            
            #blue car2 tracking
            if car2MaxVal == 1:
                if initializedCar2Tracker == 0:
                    cv.rectangle(frame, [car2MaxLoc[0]-car2W+30, car2MaxLoc[1]-2*car2H], [car2MaxLoc[0]+2*car2W+30, car2MaxLoc[1]+2*car2H], (255, 0, 0), 3 )
                    boxCar2 = [car2MaxLoc[0]-car2W+30, car2MaxLoc[1]-2*car2H, 3*car2W, 4*car2H]
                    # print(car2MaxLoc[0]-car2W+10, car2MaxLoc[1]+10, 2*car2W, 2*car2H)
                    car2Tracker.init(frame, boxCar2)
                    initializedCar2Tracker = 1
            if initializedCar2Tracker == 1:
                 retCar2, boxCar2 = car2Tracker.update(frame)
                 if retCar2:
                    p1Car2 = (int(boxCar2[0]), int(boxCar2[1]))
                    p2Car2 = (int(boxCar2[0] + boxCar2[2]), int(boxCar2[1] + boxCar2[3]))
                    cv.rectangle(frame, p1Car2, p2Car2, (255, 0, 0), 3, 1)
                    cv.putText(frame, "Car 2", [p1Car2[0], p1Car2[1]-10], 0 , 1.7, (255, 0, 0), 3)

            # person tracking
            if persMaxVal == 1:
                if initializedPersTracker == 0:
                    cv.rectangle(frame, [persMaxLoc[0]-persW, persMaxLoc[1]-persH//2], [persMaxLoc[0]+persW+20, persMaxLoc[1]+round(3/2*persH)+50], (0, 255, 0), 3 )
                    boxPers = [persMaxLoc[0]-persW, persMaxLoc[1]-persH//2, 2*persW+20, 2*persH+50]
                    # print(persMaxLoc[0]-persW+10, persMaxLoc[1]+10, 2*persW, 2*persH)
                    persTracker.init(frame, boxPers)
                    initializedPersTracker = 1
            if initializedPersTracker == 1:
                 retPers, boxPers = persTracker.update(frame)
                 if retPers:
                    p1Pers = (int(boxPers[0]), int(boxPers[1]))
                    p2Pers = (int(boxPers[0] + boxPers[2]), int(boxPers[1] + boxPers[3]))

                    #blurring part
                    # mask = np.zeros_like(frame[:, :, 0])
                    # # print("blurring")
                    # # Define the rectangular region to be blurred (x, y, width, height)
                    # x, y, w, h = int(boxPers[0]), int(boxPers[1]), int(boxPers[2]), int(boxPers[3]/2)
                    # mask[y:y+h, x:x+w] = 255
                    # # print(x, y, w, h)

                    # # Apply a blur filter to the selected region
                    # blurred = cv.blur(frame, (151, 151))
                    # frame = np.where(np.expand_dims(mask, axis=2) == 255, blurred, frame)
                    cv.rectangle(frame, p1Pers, p2Pers, (0, 255, 0), 3, 1)
                    cv.putText(frame, "Worker", [p1Pers[0], p1Pers[1]-10], 0 , 1.7, (0, 255, 0), 3)

            # bus tracking
            if busMaxVal == 1:
                if initializedBusTracker == 0:
                    cv.rectangle(frame, [busMaxLoc[0]-busW-170, busMaxLoc[1]-40], [busMaxLoc[0]+busW-170, busMaxLoc[1]+2*busH-40], (0, 0, 255), 3 )
                    boxBus = [busMaxLoc[0]-busW-170, busMaxLoc[1]-40, 2*busW, 2*busH]
                    # print(busMaxLoc[0]-busW+10, busMaxLoc[1]+10, 2*busW, 2*busH)
                    #tracker KCF
                    busTracker.init(frame, boxBus)
                    #tracker MedianFlow
                    # busTracker.init(frame, boxBus)
                    initializedBusTracker = 1
            if initializedBusTracker == 1:
                 retBus, boxBus = busTracker.update(frame)
                 if retBus:
                    p1Bus = (int(boxBus[0]), int(boxBus[1]))
                    p2Bus = (int(boxBus[0] + boxBus[2]), int(boxBus[1] + boxBus[3]))
                    cv.rectangle(frame, p1Bus, p2Bus, (0, 0, 255), 3, 1)
                    cv.putText(frame, "Bus", [p1Bus[0], p1Bus[1]-10], 0 , 1.7, (0, 0, 255), 3)
            if busMaxValReal < 284705792.0: 
                initializedBusTracker = 0
                busTracker = cv.TrackerKCF_create()

            cv.imshow("Resized_Window", frame)
            
                
            videoWriter.write(frame) 
             

            key = cv.waitKey(100)
            if key == ord('q'):
                break
    else:
        break
