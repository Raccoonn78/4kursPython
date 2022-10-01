from cgi import test
import imp
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
import seaborn as sns
import statistics as st
import statistics
import seaborn as sns
import matplotlib.ticker as ticker
import random
from cmath import pi, sqrt
import math as M

df= pd.read_csv('4kurs\TIABD\insurance2.csv' ,sep=',',  encoding='cp1251', decimal=',')
b=300 # количество интервалов разбиения на гисторгамме  
g= len(df['charges']) # количество генерирумеых значений случайной величины 
print(type(df['bmi'][10]))

def teoretical_freqeuncias(X): # теорему нашел в инете , то как она работает можно найти на youtube https://www.youtube.com/watch?v=ncNCJd4WNJ0&t=40s
    m,s =st.mean(X), st.stdev(X) # параметры нормального распределения 
    mx, mn = max(X), min(X)# граничные значения X 
    x_step = (mx-mn)/b# длина интервала X 
    ndist = st.NormalDist(mu=m, sigma=s)# создание нормального распределния 
    x = [mn + n * x_step for n in range (b) ] # начальные точки интервалов 
    x_means= [i + x_step/2 for i in x[:-1:]] # cepeðины интервалов 
    p = [ndist.cdf(x[i+1]) - ndist.cdf(x[i]) for i in range (len (x) - 1)] # вероятности nonaдaнua в интервалы 
    ft= [i * g for i in p] # теортические частоты 
    return x_means, ft




def test_funct():
    df= pd.read_csv('4kurs\TIABD\insurance2.csv' ,sep=',',  encoding='cp1251', decimal=',')###важная отметка если поставить index_col=0  то он не будет читать первый столбец.
    #print(df)
    #print(df.head())
    print(df['children'])

    print(df.describe())#Описательная статистика включает те, которые суммируют центральную тенденцию, дисперсию и форму распределения набора данных, за исключением значений NaN .

    x = np.arange(len(df[:61]))  
    width = 0.35 
    fix , ax = plt.subplots(  figsize=(40,5)  )
    rec1 = ax.bar(x - width/2, df['children'][:61], width, label='children width')
    rec2 = ax.bar(x + width/2, df['age'][:61], width, label='age width')
    #rec3 = ax.bar(x - width/2, df['charges'][:61], width, label='charges width')


    ax.set_ylabel('cm')
    ax.set_xticks(x)
    ax.legend()

    plt.show()


def func_sns_all():
    df= pd.read_csv('4kurs\TIABD\insurance2.csv' ,sep=',',  encoding='cp1251')###важная отметка если поставить index_col=0  то он не будет читать первый столбец.
    
    fix ,ax = plt.subplots(figsize=(10, 5))
    sns.despine(fix)

    sns.histplot(
        df,
        x="charges", hue="region",
        multiple="stack",
        palette="light:m_r",
        edgecolor=".3",
        linewidth=.5,
        log_scale=True,
        )
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter()) # ScalarFormatter() - Средство форматирования по умолчанию для скаляров: автовыбор строки формата.
    #set_major_formatter - Установите средство форматирования основного тикера.
    #ax.set_xticks([500, 1000, 2000, 5000, 10000])
    plt.show()

def func_sns_bmi():
    df= pd.read_csv('4kurs\TIABD\insurance2.csv' ,sep=',',  encoding='cp1251')###важная отметка если поставить index_col=0  то он не будет читать первый столбец.
    
    fix ,ax = plt.subplots(figsize=(10, 5))
    sns.despine(fix)

    sns.histplot(
        df,
        x="bmi", hue="region",
        multiple="stack",
        palette="light:g_r",
        edgecolor=".3",
        linewidth=.5,
        log_scale=True,
        )
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter()) # ScalarFormatter() - Средство форматирования по умолчанию для скаляров: автовыбор строки формата.
    #set_major_formatter - Установите средство форматирования основного тикера.
    #ax.set_xticks([500, 1000, 2000, 5000, 10000])
    plt.show()    


def func_moda():
    df= pd.read_csv('4kurs\TIABD\insurance2.csv' ,sep=',',  encoding='cp1251')###важная отметка если поставить index_col=0  то он не будет читать первый столбец.
    all_summ=0
    all_summ_charges=0
    
    for col_name,numbers in df['bmi'].items():
        
        all_summ=numbers+all_summ
        
    for col_name_charges,numbers_charges in df['charges'].items():
        
        all_summ_charges=numbers_charges+all_summ_charges


    g_mean = statistics.geometric_mean(df['bmi'])
    all_summ=all_summ/len(df['bmi'])
    garmon=len(df['bmi']) / np.sum(1.0/df['bmi'])


    g_mean_charges = statistics.geometric_mean(df['charges'])
    all_summ_charges=all_summ_charges/len(df['charges'])
    garmon_charges=len(df['bmi']) / np.sum(1.0/df['charges'])
    
    
    
    print('среднее значение колонки bmi = ',all_summ)

    print('среднее геометрическое колонки bmi = ' , g_mean)

    print('среднее гармническое колонки bmi = ', garmon)

    print('среднее значение колонки charges = ',all_summ_charges)

    print('среднее геометрическое колонки charges = ' , g_mean_charges)

    print('среднее гармническое колонки charges = ', garmon_charges)


    fix ,ax = plt.subplots(figsize=(40, 5))

    x=['среднее значение колонки bmi','среднее геометрическое колонки bmi', 'среднее гармническое колонки bmi' ]
    xx=['среднее значение колонки charges', 'среднее геометрическое колонки charges',  'среднее гармническое колонки charges']
    y=[all_summ,g_mean,garmon]
    yy=[all_summ_charges,g_mean_charges,garmon_charges]
    plt.subplot(1, 2, 1)
    sns.barplot(x=x,y=y,  hue=x)
    plt.subplot(1, 2, 2)
    sns.barplot(x=xx,y=yy, hue=xx)
    
    plt.show()





    pass

    
def boxplot():
    df = pd.read_csv('4kurs\\TIABD\\insurance2.csv', sep=',',encoding='cp1251', decimal=',')
    age=df['age']
    children=df['children']
    bmi=df['bmi']
    charges=df['charges']
    
    
    df = pd.DataFrame(dict(
        age=age,
        charges=charges,
        children=children,
        bmi=bmi
                )).melt(var_name="quartilemethod")


    fig = px.box(df, y="value", facet_col="quartilemethod", color="quartilemethod",
             boxmode="overlay", points='all')

    fig.update_traces( jitter=0, col=1)
    fig.update_traces( jitter=0, col=2)
    fig.update_traces( jitter=0, col=3)
    fig.update_traces( jitter=0, col=4)

    fig.show()



def charges_teoretical():
    df= pd.read_csv('4kurs\TIABD\insurance2.csv' ,sep=',',  encoding='cp1251', decimal=',')
    charges= df['charges']
    new_l = list(map(float, df['bmi']))
    pd = pd.DataFrame(new_l)## дисперчия
    sample = np.random.normal(loc=500, scale=150, size=300)
    print(sample.mean(), sample.std())

    fig, axes = plt.subplots(figsize = (7, 4))
    sns.histplot(data=new_l , ax=axes)
    axes.set_title('Изначальная выборка (n = 300)', size=12)
    dispers= pd.var()
    standart= dispers/ M.sqrt(100)
    standart1=dispers/ M.sqrt(1000)
    standart2=dispers/ M.sqrt(10000)
    standart3=dispers/ M.sqrt(100000)
    print('стандартное отклонение для 100 = ',standart)
    print('стандартное отклонение для 1000 = ',standart1)
    print('стандартное отклонение для 10000 = ',standart2)
    print('стандартное отклонение для 100000 = ',standart3)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (15, 8))
    avg = [[0, 0], [0, 0]]
    size = [[100, 1000], [10000, 100000]]
    for i in range(2):
        for j in range(2):
            
            avg[i][j] = [np.mean(random.choices(new_l , k=300)) for i in range(size[i][j])]
            
            sns.histplot(data=avg[i][j], ax=axes[i][j])
            axes[i][j].set_title('Распределение {:d} выборочных средних'.format(size[i][j]), size=12)

    plt.show()

def last_task(): 
    n=1
    new_l = list(map(float, df['bmi']))  # or just map(int, l) in Python 2 конвертируем числа str в float 
    
    second_list= list()
    for i in  new_l:
        second_list.append(i/n)
    print(new_l)
    print(second_list)
    x , ft = teoretical_freqeuncias(second_list)   # вызываем функцию для подсчета той самой теоремы
     
    # plt.hist(second_list, edgecolor= 'black', bins=b)
    # plt.plot(x, ft , color='red')
    
    sns.regplot(x=x, y=ft)
    plt.xlabel('bmi')
    plt.ylabel('Частота')

    plt.show()


    pass





# test_funct()

# func_sns_all()

# func_sns_bmi()

# func_moda()

# boxplot()


# charges_teoretical()

# last_task()



