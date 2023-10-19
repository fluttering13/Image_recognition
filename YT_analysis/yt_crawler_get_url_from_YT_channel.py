from bs4 import BeautifulSoup
from selenium import webdriver
import time
from selenium.webdriver.common.by import By
import pickle

def scroll():
    driver.execute_script("scroll(0,8000)")
    time.sleep(2)
    all_window_height =  []  # 创建一个列表，用于记录每一次拖动滚动条后页面的最大高度
    all_window_height.append(driver.execute_script("return document.documentElement.scrollHeight;")) #当前页面的最大高度加入列表
    while True:
        driver.execute_script("scroll(0,1000000)") 
        time.sleep(1)
        check_height = driver.execute_script("return document.documentElement.scrollHeight;")
        print(check_height,all_window_height[-1])
        if check_height == all_window_height[-1]:  
            break
        else:
            all_window_height.append(check_height)

target_url = "https://www.youtube.com/@user-vx5bd2fn6y/videos"

options = webdriver.ChromeOptions()
options.add_argument("headless")
#options.add_argument('user-agent="MQQBrowser/26 Mozilla/5.0 (Linux; U; Android 2.3.7; zh-cn; MB200 Build/GRJ22; CyanogenMod-7) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1"')
options.add_argument('blink-settings=imagesEnabled=false') 
options.add_argument('--no-sandbox')
driver = webdriver.Chrome(options=options)
driver.get(target_url)
driver.maximize_window()

scroll()

titles_list=driver.find_elements(By.ID,'video-title-link')
titles_text_list=[]
url_list=[]
for obj in titles_list:
    try:
        titles_text_list.append(obj.text)
        url_list.append(obj.get_attribute('href'))
    except:
        pass

print(titles_text_list,url_list)
print(len(titles_text_list),len(url_list))
dict_url={'url_list':url_list,'titles_text_list':titles_text_list}
fp=open('./YT_analysis/dict_url.pkl', 'wb')
pickle.dump(dict_url, fp)




