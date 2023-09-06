# 首抽腳本的建立
此處我們主要目的是刷各種麻煩的首抽，其中會用到圖形辨識與操控程式的技巧

圖像辨識的部分我們會使用到pyautogui跟CV2的包

抓圖用Grabimage

pyautigui主要是控制滑鼠跟鍵盤的程式，同時也有支援簡單的圖像辨識以配合鼠鍵操作

CV2上我們主要是使用大張的圖像辨識，有一些算法可以幫助我們縮短這個過程

＃ 前言

這個鬼東西足足花了我五天的時間，有很多時間都是在建立圖像位置跟定位程式

先有了一個雛形後再慢慢修細節，兩天都在無限重跑繼續抓蟲

例如滑鼠擋到圖標會導致預測結果失真，跟每張圖片用的準確率還要再抓再側

會依照關卡的難度不同，而有不同的難度跟除蟲，跟很多時候有很難掌控的因素

後續試跑又有一些神奇難以預測的BUG，總之不容易

# CV2上手

OpenCV是一個大型的包可以用來處理影像或是圖片的辨識工作

## CV2安裝
```
pip install opencv-python
```
## 基本code
我們導入cv2包，再讀取圖片
```
import cv2
img_path='./your image path'
img=cv2.imread(img_path)
```
通常（x,y,3）RGB的格式，也支援以下function

IMREAD_UNCHANGE:包含透明度之類的全部的dimension

IMREAD_GREYSCALE:灰階就只有二維的(X,Y)

IMREAD_COLOR:RGB三色的(X,Y,3)

此處我們只做灰階也足夠，但是測試算法的時候使用RGB涵蓋的資訊比較多，也通常會比較準

## 顯示圖片
以下是怎麼顯示圖片的例子
```
cv2.imshow('window's name', img)
cv2.delay(0) ## stop the window vanishing
cv2.destoryWindow('window's name') ## release after displying
```
cv2本身有支援matplot
```
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()
```
## 影像辨識
內部有些好用的影像辨識的function，從原理跟實際應用好好介紹一下
### 單模辨識
```
matchTemplate(InputArray image, InputArray templ, method)
```
method裡頭提供六種function

以下演示如何使用對tmpl的搜索

```
    #模板匹配
    result = cv2.matchTemplate(img_src, img_templ, method)
    print('result.shape:',result.shape)
    print('result.dtype:',result.dtype)
    #计算匹配位置
    min_max = cv2.minMaxLoc(result)
    if method == 0 or method == 1:   #根据不同的模式最佳匹配位置取值方法不同
        match_loc = min_max[2]
    else:
        match_loc = min_max[3]      
    #注意计算右下角坐标时x坐标要加模板图像shape[1]表示的宽度，y坐标加高度
    right_bottom = (match_loc[0] + img_templ.shape[1], match_loc[1] + img_templ.shape[0])
    print('result.min_max:',min_max)
    print('match_loc:',match_loc)
    print('right_bottom',right_bottom)
    #标注位置
    img_disp = img_src.copy()
    cv2.rectangle(img_disp, match_loc,right_bottom, (0,255,0), 5, 8, 0 )
    cv2.normalize( result, result, 0, 255, cv2.NORM_MINMAX, -1 )
    cv2.circle(result, match_loc, 10, (255,0,0), 2 )
```
其中，result記錄著所有匹配的搜尋結果

此處再利用minMaxLoc的函數把最匹配的結果拉出來

找到的座標相當於是圖的左上角，加上tmple圖片的大小可以拿到圖片的右下角

我們可以再利用rectangle函數繪製找到的區域

### 多模辨識

有時候我們希望把符合的所有結果都標出來，這邊我們有很多種做法

方法一：我們設定一個theshold來把符合的圖片抓出來

```
val,result = cv2.threshold(result_t,0.9,1.0,cv2.THRESH_BINARY)
match_locs = cv2.findNonZero(result)
print('match_locs.shape:',match_locs.shape) 
print('match_locs:\n',match_locs)
```

假如打印出來

```
match_locs.shape: (7, 1, 2)
match_locs:
 [[[523  96]]
 [[524  96]]
 [[471 145]]
 [[471 146]]
 [[471 195]]
 [[154 244]]
 [[154 245]]]
```
找到了七個點，但事實上只有四個

我們可以再利用tmple圖片的大小，把重複的砍掉就行了


＃ 成果
策略：CODE主要分兩塊，一塊是根據圖片辨識來按按鈕，另外一塊是如果圖片辨識失敗就點預先設好的絕對位置。

PS:絕對位置不好設就先抓其他的絕對位置在抓相對位置
## 跑關
這邊應該是最花時間跟心思的，要不斷測試按鈕能不能按跟人物能不能跑到位置上

要考慮一些留白或是多餘的動作來避免最糟糕的事情發生
## 自動轉蛋與辨識UR

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Project-recongnition/main/Danchro_random_egg/pic/test.png" width="300px"/></div

首先我們建立CODE,這裡我們使用Non-Maximum Suppression 來刪除那些被判定是同個位置的圖片
```
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

```

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Project-recongnition/main/Danchro_random_egg/pic/new_img.png" width="300px"/></div

看一下結果

```
2
```
這部分是最舒服的，由於遊戲內有無限首刷，操作過程就是幾個按鈕的事情

這邊會建立辨識場景，辨識是否在抽選結果頁面，不在的話就點來跳頁

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Project-recongnition/main/Danchro_random_egg/pic/sce.png" width="300px"/></div

可以根據個人需求設置停止點，因為每個人視窗大小不同，當圖片辨識沒看到的時候就直接點螢幕

這邊要自己再打座標進去

```
def click_yes():
    try:
        buttonLocation = pyautogui.locateOnScreen(yes_path, confidence=0.9)
        button_center = pyautogui.center(buttonLocation)
        btnX, btnY = button_center
    except:
        #print('yes error')
        button_center=(977,647)
    pyautogui.moveTo(button_center, duration = 0.5)
    pyautogui.click(button_center)

def click_restart():
    try:
        buttonLocation = pyautogui.locateOnScreen(restart_path, confidence=0.8)
        button_center = pyautogui.center(buttonLocation)
        btnX, btnY = button_center
        pyautogui.moveTo(button_center, duration = 1)
        pyautogui.click(button_center)
    except:
        #print('arrow_error, use the defeaut corridinate')          
        pyautogui.moveTo(613, 842, duration = 1)
        pyautogui.click(clicks=1)

def click_skip():
    try:
        buttonLocation = pyautogui.locateOnScreen(skip_path, confidence=0.8)
        button_center = pyautogui.center(buttonLocation)
        btnX, btnY = button_center
        pyautogui.moveTo(button_center, duration = 1)
        pyautogui.click(button_center)
    except:
        #print('arrow_error, use the defeaut corridinate')          
        pyautogui.moveTo(1375, 96, duration = 1)
        pyautogui.click(clicks=1)

def all_process():
    click_restart()
    time.sleep(1)
    click_yes()
    time.sleep(1)
    click_skip()
```

4張以上：100至200重刷抽出一次

5張以上：1000次重刷抽出一次

四張是比較可以接受的範圍抽一次大約需要40秒,大概幾個小時就可以收竿一次，看是不是自己目標的角色
## Line通知
```
def send_message():
    # LINE Notify 權杖
    token = 'your_token_code'

    # 要發送的訊息
    message = '抽到了，好了！'

    # HTTP 標頭參數與資料
    headers = { "Authorization": "Bearer " + token }
    data = { 'message': message }

    # 以 requests 發送 POST 請求
    requests.post("https://notify-api.line.me/api/notify",
        headers = headers, data = data)
```
這邊是附加功能，先進到Line notify 頁面登入設置權杖
再利用以下的code就可以傳訊息給自己嚕！
<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Project-recongnition/main/Danchro_random_egg/pic/line_pic.png" width="300px"/></div
