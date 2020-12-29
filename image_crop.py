import cv2

def img_crop(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield(x, y, image[y:y+windowSize[1], x:x+windowSize[0]])
            
file_name = 'xxx.png'
img = cv2.imread(file_name)

for c,(x, y, window) in enumerate(img_crop(img, stepSize=512, windowSize=(512,512))):
    file_name = str(c) + '.png'
    cv2.imwrite(file_name, window)
