import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import jieba
import os
import re
import time
from gensim import corpora,models

def load_stopword(path="stopwords.txt"):
    #读入停用词，返回停用词列表
    with open(path, "r", encoding="utf-8") as f:
        return set([line.strip() for line in f])


def data_processor(dataDir):
    #构建语料库
    fileList=os.listdir(dataDir)#返回文件夹包含的文件名字列表
    wordbag=[]
    stopwords=load_stopword()
    for i in fileList:
        fileName=os.path.join(dataDir,i)
        with open(fileName, "r", encoding="utf-8") as f:#读取每一条新闻
            context = f.read().strip()
            segList=jieba.cut(context)
            wordbag.append([i for i in segList if i not in stopwords])
    print(len(wordbag))

    return wordbag



if __name__ == '__main__':
    data_dir = "RenMin_Daily"
    wordbag=data_processor(data_dir)#构造语料库

    dic = corpora.Dictionary(wordbag)
    # 根据字典，将每行文档都转换为索引的形式
    corpus = [dic.doc2bow(text) for text in wordbag]
    # 现在对每篇文档中的每个词都计算tf-idf值
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf=tfidf[corpus]
    lda = models.LdaModel(corpus_tfidf, id2word=dic, num_topics=9, alpha='auto')  # lda模型
    # 打印一下每篇文档被分在各个主题的概率：
    print('----------打印每篇文档被分在各个主题的概率------------')
    corpus_lda = lda[corpus_tfidf]
    for corpus_lda_ in corpus_lda:
        print(corpus_lda_)
    # 打印出来所有的主题
    print('--------打印出来所有的主题----------')
    datas = lda.print_topics(num_topics=5, num_words=10)
    for data in datas:
        print(data)
    # 打印每个主题中每






