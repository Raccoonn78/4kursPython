from cgi import test
import imp
import scipy
from scipy import stats
import timeit

import math
import numbers
from turtle import filling
import matplotlib as mpl 
import matplotlib.pyplot as plt #В результате отобразится интерактивный график в отдельном окне
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from pyparsing import col 
import seaborn as sns
import statistics as st
import statistics
import seaborn as sns
import matplotlib.ticker as ticker
import random
from cmath import nan, pi, sqrt
import math as M
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.datasets import load_digits # для данных MNIST
from sklearn.manifold import TSNE # для снижения размерности с помощью t-SNE
# Манипулирование данными
import pandas as pd # для манипулирования данными


from scipy.stats import shapiro
import scipy.stats as ss

df= pd.read_csv('4kurs\TIABD\ECDCCases.csv' ,sep=',',  encoding='cp1251', decimal=',')





def one():
    print(df)
    print(df.info())
    pass


def two():
    # «Истина» представляет отсутствующее значение, а «Ложь» означает, 
    #что значение присутствует в наборе данных. В теле цикла for метод ".value_counts ()" подсчитывает количество значений "True".
    missing_data = df.isnull()
    #print(missing_data.head())


    for column in missing_data.columns.values.tolist():
        print('column = ',column)
        print(missing_data[column].value_counts())  # выводит true кол-во nan

        print(df[column].isnull().sum()) # сумма всех nan в строке 
        num_nan=df[column].isnull().sum()
        len_c=len(df[column])
        num_nan=(num_nan*100)/len_c  # считаем процент 
        print('пропущенных срток = ',num_nan,'%')
        print(" =================================================")
        print(' ')
        #print(str(int((missing_data['geoId'].get_height())) ))
#Посмотрим - сколько пропущенных значений в каждой колонке.
    
    ############################################################################################
    #удаляем столбец где больше всего nan 
    df.drop(columns = ['geoId', 'Cumulative_number_for_14_days_of_COVID-19_cases_per_100000'], axis = 1,inplace=True)
    print(df.head())


    n = len(df['popData2019'])
    index = n // 2     # Sample with an odd number of observations
    if n % 2:
       return sorted(df['popData2019'])[index]
     # Sample with an even number of observations
    mediana=sum(sorted(df['popData2019'])[index - 1:index + 1]) / 2

### меняем значения nan на нужные
    df['countryterritoryCode']=df['countryterritoryCode'].fillna('other')
    df['popData2019']=df['popData2019'].fillna(mediana)
    print('кол-во строк с  Nan = ',df['countryterritoryCode'].isnull().sum())
    print('кол-во строк с  Nan = ',df['popData2019'].isnull().sum())

    




    pass


def three():
    df= pd.read_csv('4kurs\TIABD\ECDCCases.csv' ,sep=',',  encoding='cp1251', decimal=',')

    print(df.info())
    print('deaths==',df['deaths'][1])


    day=df['day']
    month=df['month']
    year=df['year']
    cases=df['cases']
    deaths=df['deaths']
    popData2019=df['popData2019']
    
    
    df = pd.DataFrame(dict(
        day=day,
    month=month,
    year=year,
    cases=cases,
    deaths=deaths,
    popData2019=popData2019
                )).melt(var_name="quartilemethod")


    fig = px.box(df, y="value", facet_col="quartilemethod", color="quartilemethod",
             boxmode="overlay", points='all')

    fig.update_traces( jitter=0, col=1)
    fig.update_traces( jitter=0, col=2)
    fig.update_traces( jitter=0, col=3)
    fig.update_traces( jitter=0, col=4)
    fig.update_traces( jitter=0, col=5)
    fig.update_traces( jitter=0, col=6)

    #fig.show()

    
    pass
    

def three_second():
    df= pd.read_csv('4kurs\TIABD\ECDCCases.csv' ,sep=',',  encoding='cp1251', decimal=',')

    sum=0
    i=1
    for i in range(61904):
        if df['deaths'][i]>=3000:
            print("=================================================================================")
            print('day -->',df['dateRep'][i],'deaths -->',df['deaths'][i], '     country -->',df['countriesAndTerritories'][i])
            sum=sum+1
            print(" ")
    print('сумма всех дней когда сметри привысили 3000 = ',sum)
    pass

def four():
    Dup_Rows = df[df.duplicated()]
    print("\n\nDuplicate Rows : \n {}".format(Dup_Rows))

    DF_RM_DUP = df.drop_duplicates(keep='first')

    print('\n\nРезультирующий кадр данных после удаления дубликата :\n', DF_RM_DUP.head())
    pass




def five():
    ddf=pd.read_csv('C:\\Users\\Дмитрий\\Desktop\\VS_Code\\4kurs\\TIABD\\bmi.csv' ,sep=',',  encoding='cp1251', decimal=',')
    ddf2=pd.read_csv('C:\\Users\\Дмитрий\\Desktop\\VS_Code\\4kurs\\TIABD\\bmi2.csv' ,sep=',',  encoding='cp1251', decimal=',')
    print(len(ddf))
    stat, p = shapiro(ddf['bmi'].astype('float'))# тест Шапиро-Уилк
    stat2, p2 = shapiro(ddf2['bmi'].astype('float')) # тест Шапиро-Уилк
    
    print('Statistics=%.3f, p-value=%.3f' % (stat, p))
    print('Statistics=%.3f, p-value=%.3f' % (stat2, p2))

    alpha = 0.05
    if p > alpha:
        print('1===','Принять гипотезу о нормальности')
    else:
        print('1===','Отклонить гипотезу о нормальности')
    if p2 > alpha:
        print('2===','Принять гипотезу о нормальности')
    else:
        print('2===','Отклонить гипотезу о нормальности')
        
    print(ss.bartlett(ddf['bmi'].astype('float'),ddf2['bmi'].astype('float')))    
    print(scipy.stats.ttest_ind(ddf['bmi'].astype('float'),ddf2['bmi'].astype('float')))    

def six():
    list=[]
    for i in range(97):
        list.append(1)
    for i in range(98):
        list.append(2)
    for i in range(109):
        list.append(3)
    for i in range(95):
        list.append(4)
    for i in range(97):
        list.append(5)
    for i in range(104):
        list.append(6)
    
    print(scipy.stats.chisquare(list))
    pass

def seven():
    data = pd.DataFrame({'Женат': [89,17,11,43,22,1],
                   'Гражданский брак': [80,22,20,35,6,4],
                    'Не состоит в отношениях': [35,44,35,6,8,22]})
    data.index = ['Полный рабочий день','Частичная занятость','Временно не работает','На домохозяйстве','На пенсии','Учёба']
    print(data)
    print(scipy.stats.chi2_contingency(data))

    reg_plot = px.reg_plot(x='Женат', y='Не состоит в отношениях', fit_reg=True, data=data)
    plt.xlabel('marry')
    plt.ylabel('no_marry')
    plt.show()
    pass



# one()
#two()
#three()
#three_second()
#four()
#five()
#six()
seven()
