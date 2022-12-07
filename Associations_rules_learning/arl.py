
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np  

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split # для манипулирования данными
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
import timeit


# библиотека для Aprori вариант первый 

from apriori_python import apriori
from apyori import apriori


df= pd.read_csv('C:\\Users\\Дмитрий\\Desktop\\VS_Code\\4kurs\\TIABD\\Associations_rules_learning\\Market_Basket_Optimisation.csv')

df2= pd.read_csv('C:\\Users\\Дмитрий\\Desktop\\VS_Code\\4kurs\\TIABD\\Associations_rules_learning\\data.csv')



def first():

   
    for i in range(2):
        if i==0:
            df.stack()[:21].value_counts(normalize=True).plot(kind='bar')
            #plt.show()
        else:
            df.stack()[:21].value_counts(normalize=True).apply(lambda item: item/df.shape[0]).plot(kind='bar') # выводим первые 20 элементов 
            #plt.show()

def firs_algoritm():
    transactions=[]

    for i in range(df.shape[0]):
        row = df.iloc[i].dropna().tolist() 
        transactions.append(row)

    #transactions.append( df.iloc[p].dropna().tolist() for p in range(df.shape[0]))
    t=[] 
    start = timeit.default_timer()
    t1, rules= apriori(transactions,minSup=0.1, minConf=0.6)
    len_list=[]
    """"
    for j in range(1,10):
        minS=j/100
        for k in range(1,10):
            minC=k/10
            start1 = timeit.default_timer()
            t1, rules= apriori(transactions,minSup=minS, minConf=minC)
            end1 = timeit.default_timer()
            time_ckl=str(end1-start1)
            len_list.append(len(rules))
            app_all='minSup='+str(minS)+' munConf='+str(minC)+' Время'+time_ckl
            t.append(app_all)
    """
    end = timeit.default_timer()
    time_one=str(end-start)
    t.append(time_one)
    #print('result',rules,t1)
    #print(len_list)


    start = timeit.default_timer()
    rules=apriori(transactions=transactions, min_support=0.1,min_confidence=0.6, min_lift=1.0001)
    result=list(rules)
    end = timeit.default_timer()
    time_one=str(end-start)
    t.append(time_one)
    print(result)
    


    
    





def second():

    for i in range(2):
        if i==0:
            df2.stack()[:21].value_counts(normalize=True).plot(kind='bar')
            plt.show()
        else:
            df2.stack()[:21].value_counts(normalize=True).apply(lambda item: item/df.shape[0]).plot(kind='bar') # выводим первые 20 элементов 
            plt.show()

    transaction=[]
    for i in range(df2.shape[0]):
        rwo=df2.iloc[i].dropna().tolist()
        transaction.append(rwo)

    start = timeit.default_timer()






    end = timeit.default_timer()
    time_one=str(end-start)



firs_algoritm()