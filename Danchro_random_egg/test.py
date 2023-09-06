import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt

file_path='./Danchro_random_egg/pic/test.png'
ur_path='./Danchro_random_egg/pic/ur.png'
new_img_path='./Danchro_random_egg/pic/new_img.png'
screen = cv2.imread(file_path)
template =  cv2.imread(ur_path)
image_x, image_y = template.shape[:2]  
result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)

### Filter results to only good ones
threshold = 0.7 # Larger values have less, but better matches.
(yCoords, xCoords) = np.where(result >= threshold)

### Perform non-maximum suppression.
template_h, template_w = template.shape[:2]
rects = []
for (x, y) in zip(xCoords, yCoords):
    rects.append((x, y, x + template_w, y + template_h))
    pick = non_max_suppression(np.array(rects))
# Optional: Visualize the results

for (startX, startY, endX, endY) in pick:
    new_img=cv2.rectangle(screen, (startX, startY), (endX, endY),(0, 255, 0), 2)
    obj_number=len(pick)

#new_img.save(new_img_path)
cv2.imshow('My Image', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()