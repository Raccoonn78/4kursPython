
from sklearn.datasets import fetch_california_housing
import numpy as np 
import pandas as pd
#x, y = fetch_california_housing(return_X_y=True)




def first():
    print("Первое задание")
    summ=0
    summX=0
    while summ==0:
        numm = int(input("Введите число: "))
        summ=summ+numm
        summX=summX+numm**2
    print(summX,"== Сумма квадратов всех чисел")


def second():
    print("Второе задание")
    num = int(input('Введите число: '))
    key =0
    numb=0
    for i in range(1,num):
        numb+=1
        for j in range(i):
            
            if key==num:
                return 
            else:
                print(numb, end=' ')
            key+=1
    pass
    
    

def three():
    print("Третье задание")
    matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
            ]
 
    flat_matrix = [num for row in matrix for num in row]    
    print(flat_matrix)

def four():
    print("Четвертое задание")
    dict_new={}
    a = [ 1,   2,   3,   4,   2,   1,   3,   4,   5,   6,   5,   4,   3,   2]
    b = ['a', 'b', 'c', 'c', 'c', 'b', 'a', 'c', 'a', 'a', 'b', 'c', 'b', 'a']
    for i in range(len(a)):
        dict_new[b[i]]= (a[i]+dict_new[b[i]])  if (b[i] in dict_new) else (a[i])
        
    print(dict_new)
    return dict_new


def five():
    df = fetch_california_housing() # передаем в переменную все значения 
    #print(df)
    calf_hous_df_data = pd.DataFrame(data= df.data, columns=df.feature_names)# записываем в переменную значения со всеми значениями домов(data) И название колонок (feature_names))
    calf_hous_df_target= pd.DataFrame(data= df.target, columns=df.target_names)# записываем в переменную значения медианной цены дома (target) и название колонки (target_names)

    print()
    print('выводит четыре значения с названиями колонок')
    print()
    print("ID", calf_hous_df_data.sample(4))# выводит четыре значения с названиями колонок(data)(правда я хуй знает по какому принципу )
    print("ID", calf_hous_df_target.sample(4))# тоже самое что и выше только target 
    print("ID", pd.concat([calf_hous_df_data, calf_hous_df_target],axis=1))#соединение двух таблиц в одну 


    data_and_target= pd.concat([calf_hous_df_data, calf_hous_df_target],axis=1)


                                             ##относится к заданию семь 7.
    print(calf_hous_df_data.info())##относится к заданию шесть . info() работает только в DataFrame (но это не точно)
    
    
                                             ##относится к заданию восемь 8.
    print(calf_hous_df_data.isna().sum())#есть ли пропущенные значения(Nan) в data
    print(calf_hous_df_target.isna().sum())#есть ли пропущенные значения(Nan) в target
    
    
                                             ##относится к заданию девять 9.
    mask = data_and_target['HouseAge'].values >50 # задаем условие для поиска 
    mask2 = data_and_target.loc[mask] # создаем таблицу с условием и по сути можем ее вывести 
    mask3= mask2['Population'].values >2500  #задаем условие для поиска  уже в той таблице с первым условием 
    
    df_new = mask2.loc[mask3] # выводим табилцу со вторым условием 
    print(df_new) # выводим полученную таблицу с двумя условаиями 
    
    
                                            #относится к заданию десять 10.
    print(calf_hous_df_target.max())# максимальное значение
    print(calf_hous_df_target.min())# минимальное значение
    
    
    print()
    print(calf_hous_df_data.apply(np.floor, axis=1, args=(1)))## попробовать вывести одну строку со всеми аргументами 



first()
#second()
#three()
#four()
#five()


# функция для apply() задание №11
def f(x):
    mean = x.mean()
    print("==")
    print(x.name , mean )
    return mean
    

df = fetch_california_housing()
calf_hous_df_data = pd.DataFrame(data= df.data, columns=df.feature_names)

#print(calf_hous_df_data.apply(f, axis=0))

