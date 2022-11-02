import cv2
import numpy as np
from collections import deque

# Sonradan değiştebilmek için değişkene eşitleme
buffer_size = 16
pts = deque(maxlen = buffer_size)

# Mavi renk eşikleri HSV formatında / en kolay yolu paint ile bulunabilir
MaviAltSinir = (84,  98,  0)
MaviUstSinir = (179, 255, 255)

# Video yakalama, PC kamerası , Piksel aralıkları
dataVideo = cv2.VideoCapture(0)
dataVideo.set(3,960)
dataVideo.set(4,480)

while True:
    
    success, imgOriginal = dataVideo.read()
    
    if success: 
        
        # Gauss Bulanıklaştırması
        gBlur1 = cv2.GaussianBlur(imgOriginal, (11,11), 0) 
        
        # BGR den HSV formatına
        HSV = cv2.cvtColor(gBlur1, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV Image",HSV)
        
        # Mavi için maske oluştur
        maske = cv2.inRange(HSV, MaviAltSinir, MaviUstSinir)
        cv2.imshow("mask Image",maske)
        # Maskeyi filtreleyip gürültülerden kurtulma
        maske = cv2.erode(maske, None, iterations = 2)
        maske = cv2.dilate(maske, None, iterations = 2)
        cv2.imshow("Mask + erozyon ve genisleme",maske)
        


        # kontur
        (contours,_) = cv2.findContours(maske.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(contours) > 0:
            
            # en buyuk konturu al
            c = max(contours, key = cv2.contourArea)
            
            # dikdörtgene çevir 
            rect = cv2.minAreaRect(c)
            
            ((x,y), (width,height), rotation) = rect
            
            string = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
            print(string)
            
            # Sonra kullanabilmek için dikdörtgen kordinatlarını değişkene eşitleyip sistemin anlayabilmesi için 64bit integera çevirme
            Kutu = cv2.boxPoints(rect)
            Kutu = np.int64(Kutu)
            
            # Merkez bulma için moment yöntemi
            M = cv2.moments(c)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            
            # Kutuyu kullanarak konturu çizdir: sarı renkte olsun
            cv2.drawContours(imgOriginal, [Kutu], 0, (0,255,255),2)
            
            # Merkeze bir tane nokta çizelim: pembe
            cv2.circle(imgOriginal, center, 5, (255,0,255),-1)
            
            # Stringi ekrana yazdır
            cv2.putText(imgOriginal, string, (25,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
            
            
        # ( Deque ) Merkezin yeri değiştikten sonra bi süre eski merkezi göstermek için bir rememberence çizgi ...
        pts.appendleft(center)
        
        for i in range(1, len(pts)):
            
            if pts[i-1] is None or pts[i] is None: continue
        
            cv2.line(imgOriginal, pts[i-1], pts[i],(0,255,0),3) # 
            
        cv2.imshow("Orijinal Tespit",imgOriginal)

    # ( q ) basıldığında çalışmayı durdursun...
    if cv2.waitKey(1) & 0xFF == ord("q"): break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
