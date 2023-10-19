
import pickle
from transformers import AutoTokenizer, AutoModel
from opencc import OpenCC

tw2sp = OpenCC('tw2sp')
s2twp = OpenCC('s2twp')

file_path='./YT_analysis/audio_to_text/_wH5lSI5Dgw.txt'
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


def read_path_txt_to_list(path):
    result=[]
    f = open(path, "r",encoding='utf-8')
    for line in f:
        result.append(line[:-1])
    return result

all_text_list=read_path_txt_to_list(file_path)
#print(all_text_list)
tokenizer = AutoTokenizer.from_pretrained("iioSnail/ChineseBERT-for-csc", trust_remote_code=True)
model = AutoModel.from_pretrained("iioSnail/ChineseBERT-for-csc", trust_remote_code=True)


corrected_list=[]
for i in range(len(all_text_list)):
    input_text=tw2sp.convert(all_text_list[i])
    inputs = tokenizer(input_text, return_tensors='pt')
    output_hidden = model(**inputs).logits
    result_text=''.join(tokenizer.convert_ids_to_tokens(output_hidden.argmax(-1)[0, 1:-1]))
    corrected_list.append(s2twp.convert(result_text))
    fp=open('./YT_analysis/corrected_text/'+name+'_corrected_text_list.pkl', 'wb')
    pickle.dump(corrected_list, fp)

with open('./YT_analysis/corrected_text/'+name+'_corrected_text.txt', 'w') as fp:
    for item in corrected_list:
        fp.write("%s\n" % item)    