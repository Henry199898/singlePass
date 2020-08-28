'''最终版本'''
#-*- coding : utf-8 -*-
# coding: utf-8
import torch
import pandas as pd
from tqdm import tqdm
import sys, os 
import re
from string import digits
# import mysql.connector
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from pyhanlp import HanLP
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer ,TfidfVectorizer
from gensim import matutils
from sklearn.metrics.pairwise import cosine_similarity
# HanLP.Config.ShowTermNature = False
path = os.path.dirname(os.getcwd())
sys.path.append(path)
# from single_pass_v1 import SinglePassV1, Cluster, Document
class Document():
    def __init__(self, doc_id, features):
        self.doc_id = doc_id
        self.features = features#文本的特征，这里是(文档向量)
        # self.content = content#原文
        
class Cluster():
    def __init__(self, cluster_id, center_doc_id,document_map):
        self.cluster_id = cluster_id#簇的id，用来从map中获取这个簇的信息
        self.center_doc_id = center_doc_id#核心文档的id，用来从map红获取这个文档的信息。为了减少文档信息的备份数量，簇里只存储这个
        self.members = [center_doc_id]#簇成员的id列表。由于只遍历一遍(这是single-pass的核心竞争力之一)，不存在重复的可能，这里使用list
        self.center_vec=document_map[center_doc_id] #新建簇的时候给它赋值一个中心向量，之后每添加一个文档需要重新计算中心向量
    def add_doc(self, doc_id):
        self.members.append(doc_id)
        
#一个简单的single-pass文本聚类算法
class SinglePassV1():
    # 算法中存在两个类即为Document和Cluster     
    def __init__(self):
        self.document_map = []#存储文档信息，id-content结构。当然value也可以使用对象存储文档的其他信息。
        self.cluster_list = []#存储簇的信息，
    
    #提取文本特征(使用hanlp分词)
    def get_words_1(self, text):
        words = HanLP.segment(text)
        # 下方使用map将分词的结果用字符串类型包裹起来         
        words = list(map(str, words))
        return words
    # 英文的分词以及特征工程处理
    def get_words(self,x):
        line=''     
        x=re.sub(r'[.(),#\\*\'=!%:`?/>]',' ',str(x))     
        x=re.sub(r'[-\[\]\'`{}]',' ',x)
        x = x.translate(str.maketrans('', '', digits))
        x=x.replace("\n", " ")
        
        for j in x.split(' '):
            if j=='':
                continue
            else:
                lemmatizer = WordNetLemmatizer()
                j=lemmatizer.lemmatize(j)
                line+=j+' '
        x=word_tokenize(line.strip().lower())
        stwords=stopwords.words('english')
        clean_tokens=x[:]
        for token in x:
            if token in stwords:
                clean_tokens.remove(token)
        result=''
        for token in clean_tokens:
            if token=='':
                continue
            else:
                result+=token+' '
        return result.strip()
            
    
    #输入文档列表，进行聚类。现实中，我们遇到的文档会带有id等信息，这里为了简单，只有文本内容，所以需要生成id,一遍存取。
    def fit(self, document_list):
        #对文档进行预处理
        self.document_map=self.preprocess(document_list)
        self.clutering()
    
    # 判断一个cluster对象与一份document对象是否相似，若相似返回true，否则返回false
    # 此处的similar_1方法仅仅是使用文档的词频进行判断是否相似（比较简单，实际在disscion聚类任务中使用tf-idf或者doc2vec向量来计算）     
    def similar_1(self, cluster, document):
        # 使用set进行降重计算        
        cluster_feature = set(self.document_map[cluster.center_doc_id].features)
        document_feature = set(document.features)
#         print(cluster_feature, document_feature)
        similarity = len(cluster_feature & document_feature) / len(cluster_feature | document_feature)
        if similarity > 0.1:
            return True
        else:
            return False
#   使用tf-idf的向量来计算簇中心与新进文档之间的相似关系,返回一个最大的相似度的簇
    def get_max_similarity(self, cluster_lists, document):
        max_value = 0
        max_index = -1
#         print('vector:{}'.format(vector))
        for iter,cluster in enumerate(cluster_lists):
#             print('core:{}'.format(core))

            similarity = cosine_similarity(document.reshape(1,-1), cluster.center_vec.reshape(1,-1))
            # print(similarity[0][0])
            similarity=similarity[0][0]
            if similarity >= max_value:
                max_value = similarity
                max_index = iter
                # print("iter{},max_value{},max_index{}".format(iter,max_value,max_index))
            # print("1_iter{},max_value{},max_index{}".format(iter,max_value,max_index))
        return max_index, max_value
    
    def clutering(self,theta=0.0001,path='./similarity.txt'):
        file_1=open(path,'w')
        for iter,vec in tqdm(enumerate(self.document_map)):
#             print(doc_id, self.document_map[doc_id])
            # 判断是否为第一个前来聚类的文档，即为此时簇的个数为零 

            if len(self.cluster_list)==0:
                new_cluser_id = "c_" + str(len(self.cluster_list))
                # print(new_cluser_id)
                new_cluster = Cluster(new_cluser_id, iter,self.document_map)
                self.cluster_list.append(new_cluster)
                # print(self.document_map)
                # print(self.cluster_list[0].center_vec)
                # print(self.cluster_list[0].center_vec*2)
                # print(self.cluster_list[0].center_vec+self.cluster_list[0].center_vec)
            else:

                max_index, max_value = self.get_max_similarity(self.cluster_list, self.document_map[iter])
                # print(max_value)
                file_1.writelines("iter:{},max_index{},max_value{}\n".format(iter,max_index,max_value))
                if max_value>theta:
                        self.cluster_list[max_index].add_doc(iter)
                        # 簇中新添加了文档，需要重新计算簇中心向量 
                        tmp=len(self.cluster_list[max_index].members)
                        self.cluster_list[max_index].center_vec=(self.cluster_list[max_index].center_vec*tmp+self.document_map[iter])/(tmp+1)
                else:
                    new_cluser_id_1 = "c_" + str(len(self.cluster_list))
    #                 print(new_cluser_id)
                    new_cluster_1 = Cluster(new_cluser_id_1, iter,self.document_map)
                    self.cluster_list.append(new_cluster_1)
        file_1.close()
                
                    
    #对所有文档分词，并生成id，并且对于各个文档求出向量表示
    def preprocess(self, document_list):
        print('开始处理数据')
        corpus = []
        for iter,i in tqdm(enumerate(document_list)):
            words = self.get_words(i)
            corpus.append(words)
        # 计算tf-idf向量  
        print("get_word处理完毕")           
        # vectorizer = CountVectorizer()
        # transformer = TfidfTransformer()
        vectorizer = TfidfVectorizer(max_df=0.4, min_df=2, encoding='utf-8')  # 暂时去除该属性max_features=10
        tfidf = vectorizer.fit_transform(corpus)
        # 导出权重，到这边就实现了将文字向量化的过程，矩阵中的每一行就是一个文档的向量表示
        tfidf_weight = tfidf.toarray()
        print("处理数据完成")
        return tfidf_weight

        
    #打印所有簇的简要内容
    def show_clusters(self,file_path='./1.txt'):
        file=open(file_path,'w')
        for cluster in self.cluster_list:
            print(cluster.cluster_id, cluster.center_doc_id, cluster.members)
            file.writelines("cluster.cluster_id{},cluster.center_doc_id{}, cluster.members {}\n".format(cluster.cluster_id, cluster.center_doc_id, cluster.members))
        file.close()

if __name__ == '__main__':
#     1.使用mysql直接读取
#     mydb = mysql.connector.connect(
#       host="localhost",       # 数据库主机地址
#       user="root",    # 数据库用户名
#       passwd="root",   # 数据库密码
#       database="singlepass"
#     )
#     mycursor = mydb.cursor()
#     mycursor.execute("SELECT id, content FROM discussion_new")
#     myresult = mycursor.fetchall()

#     2.mysql存入csv中读取之后操作
    data=pd.read_csv('./discussion_new.csv',names=['id','name','num','user','date','content']) 
    data=data[:10]
    # id_1=data['id'][1:]
    # print(id_1)
    content_1=data['content'].tolist()[1:]
    # myresult=[]
    # for i in range(1,len(id_1)+1):
    #     myresult.append((int(id_1[i]),content_1[i]))
    # myresult.to(device)
    single_passor = SinglePassV1()
    single_passor.fit(content_1)
    single_passor.show_clusters()