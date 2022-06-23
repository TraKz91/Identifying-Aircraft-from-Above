import sys, numpy, cv2
import numpy as np

# The size of the mask use for open and close.
mask_size = 9

# The font we'll use to write on the image.
font = cv2.FONT_HERSHEY_SIMPLEX

#read image and convert it in hsv
img=cv2.imread("Data1.jpg")
imgorg=cv2.imread("Data1.jpg")
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#color for the mask
white_1 = np.array([80,0,80],dtype=np.uint8)
white_2 = np.array([130,100,255],dtype=np.uint8)

#create the mask and apply morphology function to have a better threshold
mask=cv2.inRange(hsv,white_1,white_2)
l=numpy.ones ((5, 5), numpy.uint8)
ef = cv2.morphologyEx (mask, cv2.MORPH_OPEN, l)
el=cv2.morphologyEx (ef, cv2.MORPH_CLOSE, l)
kernel1=numpy.ones ((7, 7), numpy.uint8)
erosion = cv2.erode(el,kernel1,iterations = 1)
dilation = cv2.dilate(erosion,kernel1,iterations = 1)
kernel2=numpy.ones ((9, 9), numpy.uint8)
dilation1 = cv2.dilate(dilation,kernel2,iterations = 2)
erosion1 = cv2.erode(dilation1,kernel2,iterations = 2)

#apply the mask
res=cv2.bitwise_and(img,img,mask= erosion1)

#convert the image into gray
rgb = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

#threshold the image
blur = cv2.GaussianBlur(gray,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#find the contours
contours, bim = cv2.findContours (th3, cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
"""
cv2.drawContours (img, contours, -1, (255,255,0), 5)


cv2.drawContours (img, contours, -1, (255,255,0), 5)

a=0
for i in range (0,len(contours)):
    a=a+1
print(a)

for i in range (0,len(contours)):
    rect = cv2.minAreaRect(contours[i])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,(0,0,255),2)

    x,y,w,h = cv2.boundingRect(contours[i])
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
"""

""
#loop through the contours
a=0
b=0
for i in range (0,len(contours)):
    a=a+1
    #if the are is over 1500, we draw a bounding box around the area and crop it
    area = cv2.contourArea(contours[i])
    if area>1500:   #3000
        x,y,w,h = cv2.boundingRect(contours[i])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #cv2.drawContours(img,[box],0,(0,0,255),5)
        crop = imgorg[y:y+h, x:x+w]
        cv2.imwrite('data6'+str(i)+'.jpg',crop)
        #cv2.drawContours (img, contours[i], -1, (255,255,0), 5)
        b=b+1
    #print(a)
    #print(area)
    else:
        continue

"""
nb=len(contours)
#cv2.drawContours (img, contours, -1, (255,255,0), 2)
nb=len(contours)
#print(nb)
#print(b)
#cv2.putText (img, str(nb), (10, 45), font, 1, 0, 2)
cv2.rectangle(res, (0-1, 0+1), (20+1, 20-1), (255, 255, 255), 2)


contours, bim = cv2.findContours (res, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours (img, contours, -1, (255,255,0), 2)
nb=len(contours)
print(nb)"""
"cv2.putText (img, str(nb), (10, 45), font, 1, 0, 2, cv2.LINEAA)"
cv2.imshow ("Data6Colour.jpg",img)
#cv2.imshow ("b.jpg",res)
#cv2.imwrite('test'+str(i)+''+str(j)+''+str(k)+'.jpg', yes)
#cv2.imwrite("classif.jpg",img)
cv2.waitKey(0)
cv2.destroyAllWindows ()
