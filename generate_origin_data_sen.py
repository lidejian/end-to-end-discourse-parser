import json
###PDTB四分类
dict_sense_to_label4 = {
    'Temporal':3,
    'Temporal.Asynchronous':3,
    'Temporal.Asynchronous.Precedence': 3,
    'Temporal.Asynchronous.Succession': 3,
    'Temporal.Synchrony': 3,
    'Contingency':1,
    'Contingency.Cause':1,
    'Contingency.Cause.Reason': 1,
    'Contingency.Cause.Result': 1,
    'Contingency.Condition': 1,
    'Comparison':0,
    'Comparison.Contrast': 0,
    'Comparison.Concession': 0,
    'Expansion':2,
    'Expansion.Conjunction': 2,
    'Expansion.Instantiation': 2,
    'Expansion.Restatement': 2,
    'Expansion.Alternative': 2,
    'Expansion.Alternative.Chosen alternative': 2,
    'Expansion.Exception': 2,
    'EntRel': 4,
}
cnt = 0
cnt_same=0
with open('relations_格式化.json',"r", encoding='UTF-8') as f,\
        open('parses.json',"r", encoding='UTF-8') as fparser,\
        open('sen1_ex','w', encoding='UTF-8') as fsen1_ex,\
        open('sen2_ex','w', encoding='UTF-8') as fsen2_ex,\
        open('label_ex','w', encoding='UTF-8') as flabel_ex,\
        open('sense_ex','w', encoding='UTF-8') as fsense_ex,\
        open('sen1_im','w', encoding='UTF-8') as fsen1_im,\
        open('sen2_im','w', encoding='UTF-8') as fsen2_im,\
        open('label_im','w', encoding='UTF-8') as flabel_im,\
        open('sense_im','w', encoding='UTF-8') as fsense_im:
    jrelation=json.load(f)
    jparser=json.load(fparser)
    docid=0
    for r in jrelation:
        print(cnt,'\t', r['ID'], end='\t')
        cnt+=1
        
        
        ### 打开原文
        if docid!=r['DocID']:
#             if docid!=0:
#                 raw.close()
#             raw =  open('./raw/%s'%r['DocID'],"r", encoding='UTF-8', errors='ignore')
            print(r['DocID'])
            docid=r['DocID']
        
        
        ########获取arg1、arg2所在句子编号集合arg1_sen   begin #########
        arg1Tok=r['Arg1']['TokenList']
        arg2Tok=r['Arg2']['TokenList']
        arg1_sen=set()
        for j in arg1Tok:
            arg1_sen.add(j[3])
#             print(j[3]) #每个单词所在句子编号
#         print(arg1_sen, end='')
    
        arg2_sen=set()
        for j in arg2Tok:
            arg2_sen.add(j[3])
        
        arg1_sen = list(arg1_sen) #将句子编号集合转为列表
        arg2_sen = list(arg2_sen)
        print(arg1_sen,arg2_sen)
        ########获取arg1、arg2所在句子编号集合arg1_sen   end #########
        
        
        # 标记arg1和arg2所在相同句子 same_flag
        same_flag=0
        if arg1_sen == arg2_sen:
            print('same')
            same_flag=1
            cnt_same+=1
#         #处理同一句子情况
#         if same_flag==1:
            arg1_end_num=r['Arg1']['TokenList'][-1][4]
        
        
        # 从parser里获取句子sen1
        sen1=''
        for i in arg1_sen:
#             print(jparser[docid]['sentences'][i]['words'])
            wordlist = jparser[docid]['sentences'][i]['words']
            if same_flag==1:
                wordlist = wordlist[:arg1_end_num+1]
            for word in wordlist:
#                 print(word[0], end=' ')
                sen1+=word[0]+' '
        print('arg1_sen:',sen1)
        
        sen2=''
        for i in arg2_sen:
#             print(jparser[docid]['sentences'][i]['words'])
            wordlist = jparser[docid]['sentences'][i]['words']
            if same_flag==1:
                wordlist = wordlist[arg1_end_num+1:]
            for word in wordlist:
#                 print(word[0], end=' ')
                sen2+=word[0]+' '
        print('arg2_sen:',sen2)
        
        
        ## 写入文件
        sense=r['Sense'][0]
        ty=r['Type']
        
        if r['Type'] == "Explicit":
            fsen1_ex.write(sen1+'\n')
            fsen2_ex.write(sen2+'\n')
            flabel_ex.write(str(dict_sense_to_label4[sense])+'\n')
            fsense_ex.write(sense+'\n')
        elif r['Type'] != "EntRel":
            fsen1_im.write(sen1+'\n')
            fsen2_im.write(sen2+'\n')
            flabel_im.write(str(dict_sense_to_label4[sense])+'\n')
            fsense_im.write(sense+'\n')

        
        
        
print(cnt,cnt_same,cnt_same/cnt,'sssssssssssssssuccess!!!!!!!!')