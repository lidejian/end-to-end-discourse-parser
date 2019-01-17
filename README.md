# end-to-end-discourse-parser
### 环境
python 3.6.1  
tensorflow 1.12.0-rc1

### 目录结构
└─data  
    ├─en  
    ├─gen_en   
    │  ├─exp  
    │  │  ├─blind_test  
    │  │  ├─dev  
    │  │  ├─test  
    │  │  └─train  
    │  └─imp  
    │      ├─blind_test  
    │      ├─dev  
    │      ├─test  
    │      └─train  
    ├─gen_zh  
    │  ├─exp  
    │  │  ├─blind_test  
    │  │  ├─dev  
    │  │  ├─test  
    │  │  └─train  
    │  └─imp  
    │      ├─blind_test  
    │      ├─dev  
    │      ├─test  
    │      └─train  
    └─zh  

* data 数据文件
	* en: PDTB原始文件，其中raw为原文文件夹，parser_格式化 为标注的语篇解析
	* zh: CDTB原始文件，同en
	* gen_en: 对PDTB进行生成的句子对以及语篇关系，利用generate_origin_data.py在en和zh每个数据集下生成。
		* exp: 显性语篇关系
		* imp: 隐性语篇关系
			* blind_text: 盲测集
			* dev: 开发集
			* test: 测试集
			* train: 训练集
				* sentence: karg1和arg2组成的句子对
				* sense: 每个sentence中蕴含的语篇关系
				* label: sense的类标