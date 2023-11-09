import cv2
import numpy as np
import matplotlib.pyplot as plt

template = cv2.imread('./Detection/geometry2.jpg')
gray=cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
### separate into binary form
ret, th1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
###
cv2.imshow('th1',th1)
cv2.waitKey()
cv2.destroyAllWindows()



contours,hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
'''
mode:
cv2.RETR_EXTERNAL:只檢索外部對象
cv2.RETR_LIST：偵測所有且未建立層次結構
cv2.RETR_CCOMP：偵測所有架構，建立兩級結構
cv2.RETR_TREE：偵測所有的，建立完整的層次結構
method: 估算的方法
cv2.CHAIN_APPROX_NONE：儲存所有的對稱點
cv2.CHAIN_APPROX_SIMPLE:壓縮水平，垂直和對角線段，只留下端點。例如模擬可以用4個點編碼。
cv2.CHAIN_APPROX_TC89_L1,cv2.CHAIN_APPROX_TC89_KCOS：使用Teh-Chini鏈近似演算法
offset:（可選參數）自訂點的偏移量，格式為元組，如（-10，10）表示自訂點沿X負方向偏移10個像素點，沿Y正方向偏移10個像素點
'''
draw_img=cv2.merge((th1.copy(),th1.copy(),th1.copy()))
res=cv2.drawContours(draw_img,contours,-1,(0,0,255),3)



qube_threshold=0.01
circle_threshold=0.01
ellipse_threshold=0.01

for i in range(len(contours)):
    print(i)
    c=contours[i]
    length=cv2.arcLength(c, True)
    epsilon = 0.01 * length
    vertices = cv2.approxPolyDP(c, epsilon, True)
    M = cv2.moments(c)
    center_x=M['m10']/M['m00']
    center_y=M['m01']/M['m00']
    center=[[center_x,center_y]]
    center_int=(int(center_x),int(center_y))

    ###triangle detection
    if len(vertices)==3:
        print('triangle')
        cv2.putText(draw_img, 'triangle', (int(center_x-20),int(center_y)), cv2.FONT_HERSHEY_TRIPLEX,0.5, (0, 0, 255), 1, cv2.LINE_AA)
    ###rectangle detection
    elif len(vertices)==4:
        print('rectangle')
        cv2.putText(draw_img, 'rectangle', (int(center_x-20),int(center_y)), cv2.FONT_HERSHEY_TRIPLEX,0.5, (0, 0, 255), 1, cv2.LINE_AA)
        length_list=[np.sqrt(sum(sum((vertices[i]-vertices[i+1])**2))) for i in range(len(vertices)-1)]
        length_list.append(np.sqrt(sum(sum((vertices[0]-vertices[3])**2))))
        ave=sum(length_list)/len(length_list)
        diff_list=[abs(length_list[i]-ave) for i in range(len(length_list))] 
        if sum(diff_list)/len(diff_list)/ave<qube_threshold:
            print('qube')
            cv2.putText(draw_img, 'qube', (int(center_x-20),int(center_y+20)), cv2.FONT_HERSHEY_TRIPLEX,0.5, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        ###circle detection
        ###way1: circular degree
        # ### M00 means area
        # print(length*length/(M['m00']*np.pi*4))
        ###way2: center to boundary
        length_list=[np.sqrt(sum(sum((c[i]-center)**2))) for i in range(len(c))]
        ave=sum(length_list)/len(length_list)
        diff_list=[abs(length_list[i]-ave) for i in range(len(length_list))] 
        print(sum(diff_list)/len(diff_list)/ave)
        if sum(diff_list)/len(diff_list)/ave<circle_threshold:
            print('circle')
            cv2.putText(draw_img, 'circle', (int(center_x-20),int(center_y)), cv2.FONT_HERSHEY_TRIPLEX,0.5, (0, 0, 255), 1, cv2.LINE_AA)
        ###eclipse detection
        semi_major_axis_length=max(length_list)
        semi_minor_axis_length=min(length_list)
        focal_length=np.sqrt(semi_major_axis_length**2-semi_minor_axis_length**2)

        V_1=c[np.argmax(length_list)]

        focal_1=center+(V_1-center)*(focal_length/semi_major_axis_length)
        focal_2=center-(V_1-center)*(focal_length/semi_major_axis_length)

        ellipse_radius_list=[np.sqrt(sum(sum((c[i]-focal_1)**2)))+np.sqrt(sum(sum((c[i]-focal_2)**2))) for i in range(len(c))]
        ave=sum(ellipse_radius_list)/len(ellipse_radius_list)
        diff_list=[abs(ellipse_radius_list[i]-ave) for i in range(len(length_list))] 
        if sum(diff_list)/len(diff_list)/ave<ellipse_threshold:
            print('ellipse')
            cv2.putText(draw_img, 'ellipse', (int(center_x-20),int(center_y)+20), cv2.FONT_HERSHEY_TRIPLEX,0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # cv2.circle(draw_img,(int(center[0][0]),int(center[0][1])),5,(0, 0, 255),-1)
        # cv2.circle(draw_img,(int(focal_1[0][0]),int(focal_1[0][1])),5,(0, 255, 0),-1)
        # cv2.circle(draw_img,(int(focal_2[0][0]),int(focal_2[0][1])),5,(0, 255, 0),-1)

cv2.imshow('Contours',draw_img)
cv2.waitKey()
cv2.destroyAllWindows()

# print(len(vertices))
# print(length)

# epsilon = 0.1 * cv2.arcLength(contours[5], True)


