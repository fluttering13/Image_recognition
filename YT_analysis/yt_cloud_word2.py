import re
import numpy as np
import pickle
#from transformers import BertTokenizer
#from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
#import jieba
#import jieba.analyse
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def read_path_txt_to_list(path):
    result=[]
    f = open(path, "r",encoding='utf-8')
    for line in f:
        result.append(line[:-1])
    return result

file_path='./YT_analysis/corrected_text/obqrIjodgWY_corrected_text_list.pkl'
name=''
count=0
dot_count=0
for word in file_path:
    if count==3:
        name=name+word
    if word=='/':
        count=count+1
    if word=='.':
        dot_count=dot_count+1
    if dot_count==2:
        name=name[:-1]
        break

###read the file
font_path='C:/Windows/Fonts/kaiu.ttf'
stop_words_path='./YT_analysis/stop_words_zh.txt'
stop_words_list=read_path_txt_to_list(stop_words_path)
#print(stop_words_list)
stop_words_set=set(stop_words_list)
print('finish read the stop_words')

# words_dict='./YT_analysis/userdict.txt'
# words_dict_list=read_path_txt_to_list()
# words_dict={}
# for word in words_dict_list:
#     words_dict[]=1

fp=open(file_path, 'rb')
new_data_list=pickle.load(fp)

###filter out non-chinese word
# for i in range(len(new_data_list)):
comment_list=new_data_list
for j in range(len(comment_list)):
    comment=comment_list[j]
    match_obj =re.findall(r'[\u4e00-\u9fa5]',comment)
    comment_list[j]=''.join(match_obj)


# ###load model
# # nlp task model
# # tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
# # model = AutoModelForTokenClassification.from_pretrained('ckiplab/albert-tiny-chinese-ws') # or other models above
ws = WS("./data", disable_cuda=False)

# fre_words = {}

# all_comment_list=[]
# for comments in new_data['list_comment']:
#     for one_comments in comments:
#         all_comment_list.append(one_comments)



# for i in range(0,len(new_data_list)):
#     comment_list=new_data_list[i]
#     # comment_list=all_comment_list

fre_dic={}
tf_dic={}
df_dic={}
tf_idf_dic={}
word_s = ws(comment_list,sentence_segmentation=True,segment_delimiter_set=stop_words_set)
word_s_selected=[]
count=0
for sentence in word_s:
    tmp=[]
    tmp_df_dic={}
    for word in sentence:
        if word not in stop_words_list:
            if len(word)>=2:
                tmp.append(word)
                if word in fre_dic.keys():
                    fre_dic[word]+=1
                else:
                    fre_dic[word]=1
                count=count+1
                tmp_df_dic[word]=1
    word_s_selected.append(tmp)
    for word,fre in tmp_df_dic.items():
        if word in df_dic.keys():
            df_dic[word]+=1
        else:
            df_dic[word]=1

for word,fre in fre_dic.items():
    tf_dic[word]=fre/count
    tf_idf_dic[word]=fre/count*np.log(len(word_s)/(df_dic[word]+1))

print(sorted(fre_dic.items(),key=lambda item:item[1],reverse=True))
print(sorted(tf_idf_dic.items(),key=lambda item:item[1],reverse=True))




wc = WordCloud(background_color="white",width=1000,height=1000, max_words=30,relative_scaling=0.5,normalize_plurals=False,font_path=font_path).generate_from_frequencies(fre_dic)
wc2 = WordCloud(background_color="white",width=1000,height=1000, max_words=30,relative_scaling=0.5,normalize_plurals=False,font_path=font_path).generate_from_frequencies(tf_idf_dic)
wc.to_file('./YT_analysis/word_cloud_from_vedio/bi_wc_fre_from_'+name+'.png')
wc2.to_file('./YT_analysis/word_cloud_from_vedio/bi_wc_tf_idf_from_'+name+'.png')
    # wc.to_file('./YT_analysis/word_cloud/wc_fre_from_bi_word'+str(i)+'.png')
    # wc2.to_file('./YT_analysis/word_cloud/wc_tf_idf_from_bi_word'+str(i)+'.png')

# plt.imshow(wc)
# plt.show()






# for word in word_sentence_list:
#     word=str(word)
#     if word not in stop_words_list:
#         select_list.append(word)
#         fre_words[word]+=1


###再砍stop words





###tokenize
# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# tokenizer = BertTokenizer.from_pretrained(
#     pretrained_model_name_or_path='bert-base-chinese',  # 可选，huggingface 中的预训练模型名称或路径，默认为 bert-base-chinese
#     cache_dir=None,  # 将数据保存到的本地位置，使用cache_dir 可以指定文件下载位置
#     force_download=False,   
# )

#load user-dict

# user_dic_list=[]
# with open('./YT_analysis/userdict.txt','r',encoding="utf-8") as f:
# 	for line in f:
# 		user_dic_list.append(str(line.strip('\n').split(',')))


# tokenizer.add_tokens(new_tokens=user_dic_list)


# comment_list=new_data['list_comment'][0]
# comment=comment_list[0]
# tokens = tokenizer.tokenize(comment)
# print(tokens)
