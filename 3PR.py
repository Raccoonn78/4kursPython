import matplotlib as mpl 
import matplotlib.pyplot as plt #В результате отобразится интерактивный график в отдельном окне
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px 
import plotly as py
# a = np.array([1,2,3,4,5])
# b = np.array([0,9,8,9,0])
# print(a)
#                                              https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset?resource=download
#  сайт с большим количеством баз данных записанных в excel , быстро регаешься и скачиваешь архив 
#   если при откртии excel все данные только в пермов столбце , то у excel есть специальная функция которая распределяет данные 
#  в excel -> данные -> работа с данными (текст по столбцам ), дальше просто ждем все окей и надо выбрать формат разбиения , в нашем случае через запятую и жмем готов  


df = pd.read_csv('4kurs\\TIABD\\House_Rent_Dataset.csv', sep=';', index_col=0)
#########################задание №2 ######################################################
print(df.head())
qwe=df
qwe=np.array(qwe)
print(qwe)


def second():   ##### еще это задание относится к 7му заданию потому что оно сделанно с помощью бибилотеки matplotlib

        #### так же пятое задание !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    df = pd.read_csv('4kurs\\TIABD\\House_Rent_Dataset.csv', sep=';', index_col=0)
#########################задание №2 ######################################################
    print(df.head())
    print(df.info())
    qwe=df
    qwe=np.array(qwe)
    print(qwe)
#################################################

    a=df['Size'] # в эти переменные записываем значения которые будут выведены на графике , 
    b=df['Rent']# если у вас другая база , то вмесо них пишете то что вам нужно вывести

    df=df[:10] # записывает только 10 значений , а не огромный массив данных из "тысячи" строк , можно написать вместо 10 любое другое число , для примера вы можете вывести 10 значений чтоб было проще 

    fgr=plt.figure('Вывод в отдельном окне', figsize=(10,6))

    plt.grid(True,color='azure') # включает сетку у графика            !!!!!!!!!!!!(5.3 задание  )

    plt.title('Динамика цен домов',fontsize = 30) ### заголовок

    plt.xlabel('t, дата', fontsize= 12)# то что написанно на осях X
    plt.ylabel('$, цена', fontsize= 12)# то что написанно на осях Y

    plt.xticks(np.arange(start=0, stop=len(df), step=1), rotation=17 , size=10,)# параметры для оси X
    plt.yticks(size=10)# параметтры для оси Y

    plt.legend(loc= 'lower left',# маленькая пиздюлина на графике , вообще она удобна когда у тебя несколько графиков , !!!!!!!!!!!!(5.2)
                                 # чтоб проще было опрделять что есть что 
            facecolor = 'oldlace',    #  цвет области
            edgecolor = 'r',    #  цвет крайней линии
            title = '10 домов',    #  заголовок
            title_fontsize = '15')    #  размер шрифта заголовка)   

    plt.plot(df['Date.1'],#ЗНАЧЕНИЕ ДЛЯ ОСИ x        !!!!!!!!!!!!!!!!!! задание 5.1
        df['Rent'],#ЗНАЧЕНИЕ ДЛЯ ОСИ y
        marker='.',#СТИЛЬ МЕТОК 
        color='crimson',#цвет функции
        markerfacecolor='darkblue',#основной цвет меток
        markeredgecolor= 'black',# цвет контура метки
        linewidth=2,#толщина линии функции
        markersize=3)# размер метки функции
  

    plt.show()


def three():
    df = pd.read_csv('4kurs\\TIABD\\House_Rent_Dataset.csv', sep=';', index_col=0)
    a=df['Size']
    b=df['Rent']
    df=df[:10] # записывает только 10 значений , а не огромный массив данных 
    a=a[:10]
    b=b[:10]
    
    fig, ax = plt.subplots (1,2, figsize=(12, 6)) # так и не понял почему все пишут две переменные fig и ax
    # первая диаграмма
    ax[1].bar (a, b, color = 'teal', edgecolor = 'blue')
    ax[1].grid (alpha = 0.3, zorder = 1)
    ax[1].set_title('Диаграмма')
    ax[1].set_ylabel('TWh')
    # вторая диаграмма
    ax[0].barh (a,b, color = 'teal', edgecolor = "black" ) ### можно замитить что тут не .bar а .barh это означает что график будет вертикальный
    ax[0].grid (alpha = 0.3, zorder = 1)
    ax[0].set_title('Диаграмма')
    ax[0].set_ylabel ( 'Twh')
    plt.show()


def three_second():
    df = pd.read_csv('4kurs\\TIABD\\House_Rent_Dataset.csv', sep=';', index_col=0)
    a=df['Size']
    b=df['Rent']
    name=['blue','red','green','black', 'orange', 'grey', 'purple', 'browm', 'pink', 'yellow' ]
    df=df[:10] # записывает только 10 значений , а не огромный массив данных , чтобы проще было отображать на графике в нашем примере
    a=a[:10]
    b=b[:10]
    fig = go.Figure(px.bar(x=a, y=b, color =name, text= b))# не совсем понятно почему и как распределяются цвета для отображения на графике 
    fig.update_traces(textfont_size=14, textangle=0, textposition="outside", marker=dict(line=dict(color='black', width=2)))
    fig.update_layout(
        title = 'Дома', title_font_size = 20, title_xanchor='left', title_pad_l= 700,# отступк слева в px  (title_xanchor параметр который сам ставит в нужную часть наш заголовок но почему то это не работает и я просто по пикселям делал отступ)
        xaxis_title = 'X', xaxis_title_font_size=20, xaxis_tickfont_size = 20,# параметры с xaxis относятся к оси X 
        yaxis_title = 'Y', yaxis_titlefont_size = 18, yaxis_tickfont_size = 16,legend_y=-0.1,# параметры с yaxis относятся к оси X 
        margin=dict (l=50, r=0, t=30, b=0) #поля графика
    )
    #legend_y=-0.1 делает Y координату на 10% меньше
    fig.show()
    
    




def four():
    df = pd.read_csv('4kurs\\TIABD\\House_Rent_Dataset.csv', sep=';', index_col=0)
    a=df['Size']
    b=df['Rent']
    df=df[:50] # записывает только 10 значений , а не огромный массив данных 
    a=a[:20]
    b=b[:10]
    fig = go.Figure()
    fig.update_traces(textfont_size=14, textangle=0, textposition="outside", marker=dict(line=dict(color='black', width=2)))#параметры на всей странице
    fig.add_trace(go.Pie(values=a, labels=a.index)) 
    fig.update_layout(
        title = 'Дома', title_font_size = 20, title_xanchor='left', title_pad_l= 650,# отступк слева в px 
        margin=dict (l=50, r=0, t=30, b=0)#поля графика
        
    )
    fig.show()



def six_boxplot():
    df = pd.read_csv('4kurs\\TIABD\\House_Rent_Dataset.csv', sep=';', index_col=0)
    a=df['Size']
    b=df['Rent']
    df=df[:10] # записывает только 10 значений , а не огромный массив данных 
    # a=a[:10]
    b=b[:20]
    # c=a[10:20]
    plt.boxplot(b)
    plt.grid(True,color='azure')
    plt.xlabel('Rent', fontsize= 12)
    plt.legend(loc= 'lower left',# маленькая пиздюлина на графике , вообще она удобна когда у тебя несколько графиков , !!!!!!!!!!!!(5.2)
                                 # чтоб проще было опрделять что есть что 
            facecolor = 'oldlace',    #  цвет области
            edgecolor = 'r',    #  цвет крайней линии
            title = '10 домов',    #  заголовок
            title_fontsize = '15') 
    plt.show()






#second()
#three()
#three_second()
#four()
#six_boxplot()