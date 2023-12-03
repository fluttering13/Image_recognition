# Projects-demonstration
這邊就放一些跟我覺得有趣的實作，演示一些做過的一些作品

更詳盡的內容可以在同頁面的欄目在點進去看code的細節

# Anomaly detection 實作 (AnomalyLib)

<div align=center><img src="./Detection/pic/pic1.png" width=800px"/></div>

在訓練集只有正資料集的狀況，透過一些Anomaly map的方式，抓出那些與正資料集不同的區塊

以上是實作AnomalyLib的套件，利用PADIM model抓出那些看到異常的部分


# 因果分析實戰-旅館訂房分析

<div align=center><img src="./Dowhy/Booking_cancellation/pic1.png" width=800px"/></div>

這裡是利用Dowhy因果模型套件來對實例進行因果推斷

主要可以排除混淆因子所造成的影響，抓出主要有因果關係的因素有那些

# CPGAN 壓縮隱私生成式對抗網路

<div align=center><img src="./CPGAN_example/pic/CPGAN_STRUCT.png" width="500px"/></div>

隨著大數據時代的發展，數據隱私的問題也逐漸浮現出來

如何對數據進行加密，並還能保持一定的可利用性都是一個問題

這篇實作是利用 CPGAN 來找一種對資料加密的演算法，並可以找到好的可利用性，主要是復現以下這篇文章

B. -W. Tseng and P. -Y. Wu, "Compressive Privacy Generative Adversarial Network," in IEEE Transactions on Information Forensics and Security, vol. 15



# YT_recommend_anyalysis
<div align=center><img src="./YT_analysis/wc_tf_idf_from_all_bi_word.png" width=275px"/></div>

此處以YT「反正我很閒」，觀眾的回覆與影片的內容為資訊來源

YT評論文字雲：YT評論爬蟲+文字切片+字頻分析

YT影片精華摘取-最大觀看回顧：YT連結獲取爬蟲+最大回顧爬蟲+YT影片爬蟲+信號處理+自動剪片

YT影片內容摘要：YT聲音爬蟲+(人聲強化)+聲音轉文字+(文檔糾錯)+字頻分析


# Customer-Churn
資料集來源取至Kaggle:Telco Customer Churn

從IBM服務的資料中可以看到顧客率流失率很高

從一些簡單的圖表可以做初步原因的面向探討

嚴謹一點可以使用統計檢驗與因果推斷，這邊有使用干預來排除混淆因子造成的影響

在迴歸分析方面這邊也順便做了SVM + Auto machine learning來調參


# Danchro_randown_egg
有些遊戲就是需要好一點的首抽開局才會玩得開心

很多時候刷首抽只是一些重複且無聊的動作，這邊是實作如何用pyautogui解放雙手

玩遊戲基本上就是看到什麼東西，然後我們再進行動作，這邊在利用open-cv做圖片辨識

主要是要把目標的文字抓出來再去做動作

註：這裡只有教學如何寫code

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Project-recongnition/main/Danchro_random_egg/pic/new_img.png" width="400px"/></div>

