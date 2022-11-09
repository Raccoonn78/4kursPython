
from statistics import LinearRegression, mean
from turtle import color

import matplotlib.pyplot as plt #В результате отобразится интерактивный график в отдельном окне

import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
from pyparsing import col

# Манипулирование данными

from sklearn.model_selection import train_test_split # для манипулирования данными

from sklearn.linear_model import LinearRegression

import scipy.stats as stats

import statsmodels.api as sm

import statsmodels.api as sm
from statsmodels.formula.api import ols

from scipy.stats import shapiro
import scipy.stats as ss
import scipy

# для пост-хок тестов 
from statsmodels.stats.multicomp import pairwise_tukeyhsd



def one():

    """2.	Выполнить однофакторный ANOVA тест, чтобы проверить влияние региона 
    на индекс массы тела (BMI), используя первый способ, через библиотеку Scipy"""

    df = pd.read_csv('4kurs\TIABD\insurance.csv')
    print(df.head())
    #print(df.describe())
    missing_data = df.isnull()


    ##проверка на пустые значения 

    print('=========================ПРЕДОБРАБОТКА===================================')
    for column in missing_data.columns.values.tolist():
        
        print('|                           column = ',column,' = NULL = ',df[column].isnull().sum())
    print('=========================================================================')

    mass_region_original=[]
    for i in range(len(df['region'])):
        if ((not mass_region_original)==True) or ((df['region'][i] in mass_region_original)==False):
            mass_region_original.append(df['region'][i])
            pass
    print('Уникальных регионов ==> ',mass_region_original)
    


    #генерируем случайные данные 
    region_random_chois= np.random.choice(a=mass_region_original, p=[0.24215,0.24289,0.27205,0.24291 ], size=len(df['region'])) ## получается епта что мы тут деалем ,
    # мы для каждого региона присваиваем какое кол-во чего-то ( но в нашем случае это масса тела) будем присвоино
    bmi_chois= df['bmi']

    chois_frame= pd.DataFrame({'region':region_random_chois, 'bmi':bmi_chois})
    groups= chois_frame.groupby('region').groups
    print(chois_frame.head())
    
    southwest=bmi_chois[groups['southwest']]
    southeast=bmi_chois[groups['southeast']]
    northwest=bmi_chois[groups['northwest']]
    northeast=bmi_chois[groups['northeast']]

    print('\nF_OneWay_Result =====> ',stats.f_oneway(southwest,southeast,northwest,northeast),'\n')

    
    """3.	Выполнить однофакторный ANOVA тест, чтобы проверить влияние региона на индекс
     массы тела (BMI), используя второй способ, с помощью функции anova_lm() из библиотеки statsmodels."""


    #Тренеруем нажу модель
    model = ols('bmi ~ region', data=chois_frame).fit()
    """ Как будет показано ниже, двухфакторный дисперсионный анализ (англ. two-way analysis of variance, или two-way ANOVA) позволяет установить
     одновременное влияние двух факторов, а также взаимодействие между этими факторами. 
    При наличии более двух факторов говорят о многофакторном дисперсионном анализе
     (англ. multifactor ANOVA; не путать с MANOVA - multivariate ANOVA!)"""

    anova_result= sm.stats.anova_lm(model ,type= 2)
    print(anova_result)
    

    
def two():
    """4.	 С помощью t критерия Стьюдента перебрать все пары. Определить поправку Бонферрони. Сделать выводы."""
    
    df = pd.read_csv('4kurs\TIABD\insurance.csv')
    #print(df.describe())
    #print(df.head())

    ###поисак уникальных значений region
    mass_region_original=[]
    mass_sex_original=['female','male']
    mass_smoker_origina=['yes','no']
    for i in range(len(df['region'])):
        if ((not mass_region_original)==True) or ((df['region'][i] in mass_region_original)==False):
            mass_region_original.append(df['region'][i])
            pass
    #print('Уникальных регионов ==> ',mass_region_original)
    


    #генерируем случайные данные 
    region_random_chois= np.random.choice(a=mass_region_original, p=[0.24215,0.24289,0.27205,0.24291 ], size=len(df['region'])) ## получается епта что мы тут деалем ,
    # мы для каждого региона присваиваем какое кол-во чего-то ( но в нашем случае это масса тела) будем присвоино
    sex_random_chois= np.random.choice(a=mass_sex_original, p=[0.6,0.4 ], size=len(df['sex']))
    smoker_randon_chois=np.random.choice(a=mass_smoker_origina, p=[0.452,0.548 ], size=len(df['smoker']))

    bmi_chois= df['bmi']

    chois_frame= pd.DataFrame({'region':region_random_chois, 'bmi':bmi_chois})
    groups= chois_frame.groupby('region').groups
    print(chois_frame)
    print(chois_frame.head())
    

    ##честно я пока не сильно понял как это заработало XD
    #но он вывел все пары которые от нас требовали 
    second_region_chois=[]
    for i in range(4):
        for j in range(i+1,5):
            second_region_chois.append((region_random_chois[i],region_random_chois[j]))
    

    print('Метод Бонферрони ====>', 0.05/4)
    bonfferoni=0.05/(len(second_region_chois))
    for i,j in second_region_chois:
        print('----------------------------------------------------')
        print(i,j) 
        print(scipy.stats.ttest_ind(bmi_chois[groups[i]],bmi_chois[groups[j]]))
        oo,tt=scipy.stats.ttest_ind(bmi_chois[groups[i]],bmi_chois[groups[j]])
        if bonfferoni<tt:
            print("Гипотеза принимается")
        else:
            print("Гипотеза отклоняется")
        print('----------------------------------------------------')



    """
    "a"-альфа
    Предположим, что мы применили определенный статистический критерий 3 раза
    (например, сравнили при помощи критерия Стьюдента средние значения групп А и В, А и С, и В и С)
    и получили следующие три Р-значения: 0.01, 0.02 и 0.005. Если мы хотим, чтобы групповая вероятность
    ошибки при этом не превышала определенный уровень значимости "a" = 0.05, то, согласно методу Бонферрони,
    мы должны сравнить каждое из полученных Р-значений не с "a", а с "a"/m, где m – число проверяемых гипотез.
    Деление исходного уровня значимости "a" на m – это и есть поправка Бонферрони.
    В рассматриваемом примере каждое из полученных Р-значений необходимо было бы сравнить с 0.05/3=0.017. 
    В результате мы выяснили бы, что Р-значение для второй гипотезы (0.02) превышает 0.017 и, соответственно, у нас не было бы оснований отвергнуть эту гипотезу.
    """

    
    print('Метод Бонферрони ====>', 0.05/4)
    bonfferoni=0.05/4

    #### Делаем пост-хок тесты 
    tukey = pairwise_tukeyhsd(endog=bmi_chois, groups=region_random_chois, alpha=0.05)

    tukey.plot_simultaneous()
    ##вопрос , почему мы берем именно эти значения x and y 
    plt.vlines(x=30.5 , ymin=-0.5, ymax=4.5, color= 'red')
    tukey.summary()
    plt.show()




   



def three():
    """4.	 С помощью t критерия Стьюдента перебрать все пары. Определить поправку Бонферрони. Сделать выводы."""
    
    df = pd.read_csv('4kurs\TIABD\insurance.csv')
    #print(df.describe())
    #print(df.head())

    ###поисак уникальных значений region
    mass_region_original=[]
    mass_sex_original=['female','male']
    mass_smoker_origina=['yes','no']
    northeas=0
    southwest=0
    southeast=0
    northwest=0
    for i in range(len(df['region'])):
        if df['region'][i]=='northeast':
                northeas=northeas+1
                pass
        if df['region'][i]=='southwest':
                southwest+=1
        if df['region'][i]=='southeast':
                southeast+=1
        if df['region'][i]=='northwest':
                northwest+=1
        if ((not mass_region_original)==True) or ((df['region'][i] in mass_region_original)==False):
            mass_region_original.append(df['region'][i])
            
    #print('Уникальных регионов ==> ',mass_region_original)
    print('кол-во всех примеров с регионом',len(df['region']),northeas,    southwest,  southeast,  northwest)


    #генерируем случайные данные 
    region_random_chois= np.random.choice(a=mass_region_original, p=[0.24215,0.24289,0.27205,0.24291], size=len(df['region'])) ## получается епта что мы тут деалем ,
    # мы для каждого региона присваиваем какое кол-во чего-то ( но в нашем случае это масса тела) будем присвоино
    sex_random_chois= np.random.choice(a=mass_sex_original, p=[0.6,0.4 ], size=len(df['sex']))
    smoker_randon_chois=np.random.choice(a=mass_smoker_origina, p=[0.452,0.548 ], size=len(df['smoker']))

    bmi_chois= df['bmi']

    chois_frame= pd.DataFrame({'region':region_random_chois, 'bmi':bmi_chois, 'sex':sex_random_chois,'smoker':smoker_randon_chois})
    groups= chois_frame.groupby('region').groups
    groups= chois_frame.groupby('sex').groups
    
    print(chois_frame)
    print(chois_frame.head())
    

    ##честно я пока не сильно понял как это заработало XD
    #но он вывел все пары которые от нас требовали 
    second_region_chois=[]
    for i in range(4):
        for j in range(i+1,5):
            second_region_chois.append((region_random_chois[i],region_random_chois[j]))
    

    # print('Метод Бонферрони ====>', 0.05/4)
    # bonfferoni=0.05/(len(second_region_chois))
    # for i,j in second_region_chois:
    #     print('----------------------------------------------------')
    #     print(i,j) 
    #     print(scipy.stats.ttest_ind(bmi_chois[groups[i]],bmi_chois[groups[j]]))
    #     oo,tt=scipy.stats.ttest_ind(bmi_chois[groups[i]],bmi_chois[groups[j]])
    #     if bonfferoni<tt:
    #         print("Гипотеза принимается")
    #     else:
    #         print("Гипотеза отклоняется")
    #     print('----------------------------------------------------')



    """
    "a"-альфа
    Предположим, что мы применили определенный статистический критерий 3 раза
    (например, сравнили при помощи критерия Стьюдента средние значения групп А и В, А и С, и В и С)
    и получили следующие три Р-значения: 0.01, 0.02 и 0.005. Если мы хотим, чтобы групповая вероятность
    ошибки при этом не превышала определенный уровень значимости "a" = 0.05, то, согласно методу Бонферрони,
    мы должны сравнить каждое из полученных Р-значений не с "a", а с "a"/m, где m – число проверяемых гипотез.
    Деление исходного уровня значимости "a" на m – это и есть поправка Бонферрони.
    В рассматриваемом примере каждое из полученных Р-значений необходимо было бы сравнить с 0.05/3=0.017. 
    В результате мы выяснили бы, что Р-значение для второй гипотезы (0.02) превышает 0.017 и, соответственно, у нас не было бы оснований отвергнуть эту гипотезу.
    """

    
    #print('Метод Бонферрони ====>', 0.05/4)
    bonfferoni=0.05/4

    #### Делаем пост-хок тесты 
    # tukey = pairwise_tukeyhsd(endog=bmi_chois, groups=region_random_chois, alpha=0.05)

    # tukey.plot_simultaneous()
    # ##вопрос , почему мы берем именно эти значения x and y 
    # plt.vlines(x=30.5 , ymin=-0.5, ymax=4.5, color= 'red')
    # tukey.summary()
    # plt.show()




    #Двухфакторный дисперсионный анализ с повторениями

    model_last = ols('bmi ~ C(region) + C(sex) + C(region):C(sex)', data=chois_frame).fit()
    print('Двухфакторный дисперсионный анализ с повторениями =========> \n',sm.stats.anova_lm(model_last, typ=2))

    tukey2 = pairwise_tukeyhsd(endog=bmi_chois, groups=region_random_chois, alpha=0.05)

    tukey2.plot_simultaneous()
    ##вопрос , почему мы берем именно эти значения x and y 
    plt.vlines(x=30.5 , ymin=-0.5, ymax=4.5, color= 'red')
    tukey2.summary()
    plt.show()

    pass
    pass
    
def four():
    pass



one()
two()
three()

