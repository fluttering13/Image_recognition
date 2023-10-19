from bs4 import BeautifulSoup
from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from pytube import YouTube
import pickle
# 滾動頁面
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

### get url first
fp=open('./YT_analysis/dict_url.pkl', 'rb')
urls=pickle.load(fp)
# print(urls)


dict_vedios={}

options = webdriver.ChromeOptions()
options.add_argument("headless")
#options.add_argument('user-agent="MQQBrowser/26 Mozilla/5.0 (Linux; U; Android 2.3.7; zh-cn; MB200 Build/GRJ22; CyanogenMod-7) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1"')
options.add_argument('blink-settings=imagesEnabled=false') 
options.add_argument('--no-sandbox')
driver = webdriver.Chrome(options=options)

list_title=[]
list_length=[]
list_views=[]
list_like_message=[]
list_watch_message=[]
list_comments=[]
error_list=[]
for i in range(len(urls['url_list'])):
    try:
        target_url = urls['url_list'][i]
        yt = YouTube(target_url)
        print(yt.title)           # 影片標題
        list_title.append(yt.title)
        print(yt.length)          # 影片長度 ( 秒 )
        list_length.append(yt.length)
        print(yt.author)          # 影片作者
        print(yt.channel_url)     # 影片作者頻道網址
        print(yt.thumbnail_url)   # 影片縮圖網址
        print(yt.views)           # 影片觀看數
        list_views.append(yt.views)
        driver.get(target_url)
        driver.maximize_window()


        time.sleep(2)
        like_message=driver.find_element(By.XPATH, '//*[@id="segmented-like-button"]/ytd-toggle-button-renderer/yt-button-shape/button').text
        expand=driver.find_element(By.ID,'expand').click()
        time.sleep(1)
        watch_message = driver.find_element(By.XPATH, '//*[@id="info"]/span[1]').text
        list_like_message.append(like_message)
        list_watch_message.append(watch_message)
        scroll()

        comments_section=driver.find_element(By.XPATH,'//*[@id="comments"]')

        # extract the HTML content of the comments section
        comments_html = comments_section.get_attribute('innerHTML')

        # parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(comments_html, 'html.parser')

        # extract the text of the comments
        comments = [comment.text for comment in soup.find_all('yt-formatted-string', {'class': 'style-scope ytd-comment-renderer'})]
        list_comments.append(comments)
        # # print the comments
        # print('number of comments',len(comments))
        dict_vedios={'title':list_title,'length':list_length,'views':list_views,'like_message':list_like_message,'watch_message':list_watch_message,'list_comment':list_comments}
        fp=open('./YT_analysis/dict_vedios.pkl', 'wb')
        pickle.dump(dict_vedios, fp)
    except:
        error_list.append(i)

print(error_list)


