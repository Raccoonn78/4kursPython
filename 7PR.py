from cgi import test
from dataclasses import dataclass
import imp
from statistics import LinearRegression, mean
from traceback import print_tb


import matplotlib.pyplot as plt #В результате отобразится интерактивный график в отдельном окне
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from pyparsing import col 
import seaborn as sns
# Манипулирование данными
import pandas as pd
from sklearn.model_selection import train_test_split # для манипулирования данными

from sklearn.linear_model import LinearRegression


import statsmodels.api as sm

###https://habr.com/ru/post/557998/



"""Линейная регрессия (Linear regression) — модель зависимости переменной x 
от одной или нескольких других переменных (факторов, регрессоров, независимых переменных) с линейной функцией зависимости.

Регрессия — это метод, используемый для моделирования и анализа отношений 
между переменными, а также для того, чтобы увидеть, как эти переменные 
вместе влияют на получение определенного результата."""

def one():
    
    data = {'Улица': [80,98,75,91,78],
        'Гараж': [100,82,105,89,102]
       }
  
    df2 = pd.DataFrame(data,columns= ['Улица','Гараж'])

    print("корреляция по пирсону")
    print(df2)

    street= np.array([80,98,75,91,78])
    garage= np.array([100,82,105,89,102])

    day=np.array(['Понедельник','Вторник','Среда','Четверг','Пятница'])

##Корреляции помогают увидеть характер связи между двумя непрерывными переменными
    print('переменные идеально антикоррелируют === ',df2['Улица'].corr(df2['Гараж']))

    #fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
    plt.grid(True)
    plt.xlabel('Автомобили')
    plt.ylabel('Дни недели')

    plt.scatter(street,day, marker='x')
    plt.scatter(garage, day, marker='o')



    plt.show()    
    

def two():
    data = {'Улица': [80,98,75,91,78],
        'Гараж': [100,82,105,89,102]
       }
  
    df2 = pd.DataFrame(data,columns= ['Улица','Гараж'])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x = df2['Улица'], y = df2['Гараж'])
    plt.xlabel("Livin")
    plt.ylabel("House")

    plt.show()  



def three():
    df= pd.read_csv('4kurs\\TIABD\\bitcoin.csv' )
    #print(df)
    print('')
    
    projection= 14

    df['predict']=df[['close']].shift(-projection) ##скрыли последнии 14 дней 
    #print(df)
    print('')
    
    x= df[['close']]
    y= df[['predict']]

    x= x[:-projection]
    y= y[:-projection]

    """ Таким образом вы делите свою выборку на тренировочную и тестовую часть. Обучение будет происходит 
    на тренировочной выборке, а на тестовой - проверка полученных "знаний".
    test_size используется для разбиения выборки """

    #test_size: доля выборок, если это целое число, это количество выборок.
    #random_state: это начальное число случайного числа.
    x_train,x_test, y_train, y_test = train_test_split(x,y, test_size=0.2 , random_state=42)   ### пока хуй знает что это за значений

    model = LinearRegression()##узнать как работает !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    model.fit(x_train, y_train) 

    print('наша модель точна на = ',model.score(x_test,y_test))# на сколько наша модель точна , при том что мы дали ей всего лишь один параметр  
    print('')
    print('предсказываем данные \n',model.predict(df[['close']][-projection:]) )#выбираем данные от конца до последних 14
    print('')
    




    """Одинарные и двойные квадратные скобки используются в качестве операторов индексации в языке программирования R. 
    Оба этих оператора используются для ссылки на компоненты объектов хранения R либо как на подмножество, 
    принадлежащее одному типу данных, либо как на элемент. """



## создаем новый dataframe для того чтобы нормально посмторить график
    df2 = pd.DataFrame({'predict1': x['close'] ,
        'predict2': y['predict']})


    model_second= LinearRegression()
    

    x_model= pd.DataFrame(df2['predict1'])
    y_model= pd.DataFrame(df2['predict2'])


    model_second.fit(x_model,y_model)# модель для обучения 
    

    print('Выводим коэффицент наклона нашей прямой   = ', model_second.coef_)
    print('')
    
    print(model_second.intercept_)#y-перехват
    
    print(model_second.score(x_model,y_model)) # на сколько наша модель точна


    plt.figure(figsize=(15,6))

    plt.scatter(x_model,y_model) # выводим наш график
    
    predict_model=model_second.predict(x_model)
    
    plt.plot(x_model , predict_model)# выводим нашу прямую 


    plt.xlabel('Close')
    plt.ylabel('Prediction')
    
    plt.show()
    

    


def four():
    df= pd.read_csv('4kurs\\TIABD\\housePrice.csv' )
    print(df.head())

    missing_data = df.isnull()
    #print(missing_data.head())

    #проверяем на пустые значения########раскоментить#####################################################
    """for column in missing_data.columns.values.tolist():
           print('column = ',column)
          print(missing_data[column].value_counts())
          print(df[column].isnull().sum())"""
    
    df['Address']=df['Address'].fillna('other')
    print(" ")
    print('Кол-во Nan  в столбце Address ===',df['Address'].isnull().sum(),'\n')

    #df.hist() # гистограммы
    #plt.show()######################раскоментить#####################################################

    #найдем выборочную дисперсию 
        # mean_price это выборочное среднее

    N=df['Price'].size#возьмеме кол-во наблюдений 
    mean_price= df['Price'].sum()/N # считаем выборочное среднее 

    df['Area']=pd.to_numeric(df['Area'], errors='coerce') # преобразует строковые знаяения в числовые , если попадается слово то читай докумиентацию, есть параметр ктороый просто пропускает его
    df['Price']=pd.to_numeric(df['Price'], errors='coerce')
    
    data_area=pd.to_numeric(df['Area'], errors='coerce')
    data_price=pd.to_numeric(df['Price'], errors='coerce')

    print('выборочное среднее mean_price === ',mean_price)#
    print('выборочное среднее === ',df['Price'].mean(),'\n') # так же есть встроенный метод в pandas .mean()

    var_price= 1/(N-1) * ((df['Price'] - mean_price)**2).sum() #если убрать N-1 то знаячение дудет другое , не выборочное

    print('Выборочная дисперсия var_price=== ',var_price)
    print('Выборочная дисперсия=== ',df['Price'].var(),'\n')


    """
    Ковариация — это мера того, как две случайные величины изменятся при сравнении друг с другом

    Дисперсия случайной величины – это один из основных показателей в статистике. Он отражает меру разброса данных вокруг средней арифметической.

    Коэффицент Корреляции это ковариация деленная на корень из произведения дисперсии двуз случайных велечин

    """
    mean_area=df['Area'].mean()
    var_area=df['Area'].var()

    corr_=1/(N-1)*((df['Price']- mean_price) *(df['Area'] - mean_area)).sum()*1/(np.sqrt(var_price*var_area))

    print('Корреляция == ', corr_,'\n') 

    #corr_func= df[['Price','Area']].corr()
    #print('Корреляция матрица через встроенный метод в pandas  \n', corr_func) 
    
    """Если нам надо вывести красиво матрицу корреляции то код представлен ниже
    import seaborn as sns 
    sns.heatmap( df.corr(),
                xticklabels=df.corr().columns,
                yticklabels=df.corr().columns )
    plt.show()"""
    
    plt.scatter(x=df['Area'],y=df['Price'])
    
    plt.show()
    


    lm = sm.OLS.from_formula('Area ~ Price' ,df)    
    # print('\n')
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print('++++++++++++++++coef и есть y-перехват+++++++++++++++++')
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")


    result = lm.fit()
    print(result.summary())

    coeff= df[['Area','Price']].corr().iloc[0,1]*(df['Area'].std()/df['Price'].std())
    

    print('Коэффицент y-перехват =======',coeff)
    
    sns.regplot(x=data_area, y=data_price, data=df)
    plt.show()

    pass
   
    

    
    




one()

#three()
#four()

