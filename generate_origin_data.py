import json
###PDTB###
dict_sense_to_label = {
    'Temporal':0,
    'Temporal.Asynchronous':1,
    'Temporal.Asynchronous.Precedence': 2,
    'Temporal.Asynchronous.Succession': 3,
    'Temporal.Synchrony': 4,
    'Contingency':5,
    'Contingency.Cause':6,
    'Contingency.Cause.Reason': 7,
    'Contingency.Cause.Result': 8,
    'Contingency.Condition': 9,
    'Comparison':10,
    'Comparison.Contrast': 11,
    'Comparison.Concession': 12,
    'Expansion':13,
    'Expansion.Conjunction': 14,
    'Expansion.Instantiation': 15,
    'Expansion.Restatement': 16,
    'Expansion.Alternative': 17,
    'Expansion.Alternative.Chosen alternative': 18,
    'Expansion.Exception': 19,
    'EntRel': 20,
}

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

###CDTB
# dict_sense_to_label={
#     'Causation':0,
#     'Conditional':1,
#     'Conjunction':2,
#     'Contrast':3,
#     'EntRel':4,
#     'Expansion':5,
#     'Progression':6,
#     'Purpose':7,
#     'Temporal':8,
#     'Alternative':9,
# }

# 21分类：
# ex:
#     sentence_ex
#     sense_ex
#     label_ex
# im:
#     sentence_im
#     sense_im
#     label_im

# 5分类(仅在第一层分类，分为0-4)
# im:
#     sentence_im（上）
#     sense_im（上）
#     label_im5
    
# 4分类（仅在第一层分类，不算EntRel，0-3）
# ex:
#     sentence_ex（上）
#     sense_ex（上）
#     label_ex4
# im:
#     sentence_im4
#     sense_im4
#     label_im4
    


cnt=1
with open('relations_格式化.json',"r", encoding='UTF-8') as f,\
        open('sentence_ex','w', encoding='UTF-8') as fsentence_ex,\
        open('sense_ex','w', encoding='UTF-8') as fsense_ex,\
        open('label_ex','w', encoding='UTF-8') as flabel_ex,\
        open('label_ex4','w', encoding='UTF-8') as flabel_ex4,\
        open('sentence_im','w', encoding='UTF-8') as fsentence_im,\
        open('sentence_im4','w', encoding='UTF-8') as fsentence_im4,\
        open('sense_im','w', encoding='UTF-8') as fsense_im,\
        open('sense_im4','w', encoding='UTF-8') as fsense_im4,\
        open('label_im','w', encoding='UTF-8') as flabel_im,\
        open('label_im4','w', encoding='UTF-8') as flabel_im4,\
        open('label_im5','w', encoding='UTF-8') as flabel_im5:
    jrelation=json.load(f)
    docid=0
    for r in jrelation:
        if docid!=r['DocID']:
            if docid!=0:
                raw.close()
            raw =  open('./raw/%s'%r['DocID'],"r", encoding='UTF-8', errors='ignore')
            print(r['DocID'])
            docid=r['DocID']
#         print(raw.read())
        arg1Span=r['Arg1']['CharacterSpanList']
        arg2Span=r['Arg2']['CharacterSpanList']
        sense=r['Sense'][0]
        ty=r['Type']
        print(cnt,'\t', r['ID'])
        cnt+=1
        parenthesis='###hhh#####'
        midd='######hhh####'
        
        #读取start和end
        #start为arg1开始
        #end为arg2结束
        start=arg1Span[0][0]
        if len(arg2Span)==1:
            end=arg2Span[0][1]
        else:
            end=arg2Span[1][1]

        
        #读入num1和num2
        #num1为arg1结束
        #num2位arg2开始
        num1=arg1Span[0][1]#暂时记录arg1结尾位置
        num2=arg2Span[0][0]#记录arg2开头位置
        
        if len(arg1Span)==2:# 如果有插入语，读入插入语
            raw.seek(arg1Span[0][1],0)
            if arg1Span[1][0]-arg1Span[0][1] > 10:
                parenthesis=raw.read(arg1Span[1][0]-arg1Span[0][1])
            num1=arg1Span[1][1] #更新num1
#             print('par:',parenthesis)
        if len(arg2Span)==2:
            raw.seek(arg2Span[0][1],0)
            if arg2Span[1][0]-arg2Span[0][1] > 10:
                parenthesis=raw.read(arg2Span[1][0]-arg2Span[0][1])
#             print('par:',arg2Span[0][1],arg2Span[1][0],parenthesis)
        
        #如果arg1和arg2位置反了
        if num2-num1<0:
            start,num1,num2,end=num2,end,start,num1
        
        
        #最后考虑连接词，更新start or end
        conn = r['Connective']['CharacterSpanList']
        if len(conn) ==1:
            if conn[0][0] < start:
                start = conn[0][0]
            if conn[0][1] > end:
                end=conn[0][1]
                
#         print(start,num1,num2,end,end='\t')
        
        raw.seek(start,0)
        sen=raw.read(end-start)
        sen = sen.replace(parenthesis,' ')#去除插入语
        
#         print(num2-num1,'\t',end-start,end='\t')
        
        #论元间相隔12个以上，认为不是连接词，去掉这部分。
        if num2-num1>12:
            raw.seek(num1,0)
            midd=raw.read(num2-num1)
            sen=sen.replace(midd,'.')
        
#         if num2-num1>12:
#             raw.seek(start,0)
#             sen1=raw.read(num1-start)
            
#             raw.seek(num2,0)
#             sen2=raw.read(end-num2)
            
#             sen3=sen1+'.'+sen2
            
#             sen3=sen3.replace(parenthesis,' ')
#             sen3=sen3.replace('\n','')
# #             print(sen)
# #             print('-'*20)
# #             print(sen3)
#             sen=sen3
            
#         print(len(sen))
#         print(sen)
#         print(sense)
#         print(dict_sense_to_label[sense])

        sen = sen.replace('\n','')#去除\n
        print(sen)
        if r['Type'] == "Explicit":
            fsentence_ex.write(sen+'\n')
            fsense_ex.write(sense+'\n')
            flabel_ex.write(str(dict_sense_to_label[sense])+'\n')
            flabel_ex4.write(str(dict_sense_to_label4[sense])+'\n')
        elif r['Type'] == "EntRel":
            fsentence_im.write(sen+'\n')
            fsense_im.write(sense+'\n')
            flabel_im.write(str(dict_sense_to_label[sense])+'\n')
            flabel_im5.write(str(dict_sense_to_label4[sense])+'\n')
        else:
            fsentence_im.write(sen+'\n')
            fsentence_im4.write(sen+'\n')
            fsense_im.write(sense+'\n')
            fsense_im4.write(sense+'\n')
            flabel_im.write(str(dict_sense_to_label[sense])+'\n')
            flabel_im4.write(str(dict_sense_to_label4[sense])+'\n')
            flabel_im5.write(str(dict_sense_to_label4[sense])+'\n')
print('------------------seccess!')