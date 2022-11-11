 #В результате отобразится интерактивный график в отдельном окне
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy
# для пост-хок тестов 
from statsmodels.stats.multicomp import pairwise_tukeyhsd



def one():

    """
    2.	Выполнить однофакторный ANOVA тест, чтобы проверить влияние региона 
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

    #выводим уникальные регионы
    mass_region_original=[]
    for i in range(len(df['region'])):
        if ((not mass_region_original)==True) or ((df['region'][i] in mass_region_original)==False):
            mass_region_original.append(df['region'][i])
            pass
    print('Уникальных регионов ==> ',mass_region_original)
    

    region_chois=df['region']
    bmi_chois= df['bmi']

    #создаем dataframe и помещаяем туда значения для того чтобы их сгруппировать 
    chois_frame= pd.DataFrame({'region':region_chois, 'bmi':bmi_chois})
    groups= chois_frame.groupby('region').groups
    
    print(chois_frame.head())
    
    southwest=bmi_chois[groups['southwest']]
    southeast=bmi_chois[groups['southeast']]
    northwest=bmi_chois[groups['northwest']]
    northeast=bmi_chois[groups['northeast']]

    #выводим однофакторный ANOVA тест первым способом 
    print('\nF_OneWay_Result =====> ',stats.f_oneway(southwest,southeast,northwest,northeast),'\n')

    
    """
    3.	Выполнить однофакторный ANOVA тест, чтобы проверить влияние региона на индекс
     массы тела (BMI), используя второй способ, с помощью функции anova_lm() из библиотеки statsmodels."""
    #Тренеруем нажу модель
    model = ols('bmi ~ region', data=chois_frame).fit()
    """
    Как будет показано ниже, двухфакторный дисперсионный анализ (англ. two-way analysis of variance, или two-way ANOVA) позволяет установить
     одновременное влияние двух факторов, а также взаимодействие между этими факторами. 
    При наличии более двух факторов говорят о многофакторном дисперсионном анализе
     (англ. multifactor ANOVA; не путать с MANOVA - multivariate ANOVA!)"""
    #выводим однофакторный ANOVA тест вторым способом 
    anova_result= sm.stats.anova_lm(model ,type= 2)
    print(anova_result)
    
    """
    Дисперсия – характеристика рассеивания данных вокруг их среднего значения. 
    Дисперсионный анализ (ANOVA) – статистическая процедура, используемая для сравнения средних значений определенной переменной в двух и более независимых группах. 
    Основная статистика в дисперсионном анализе – F-отношение, используемое для выявления статистической значимости различий между группами. 

    ANOVA нашел различие, поскольку p-значение меньше 
    0,05. Это означает, что фактор раса оказывает статистически значимое влияние на возраст избирателей,
     но было бы интересно узнать в каких именно группах есть влияние"""
    
def two():
    """
    4.	 С помощью t критерия Стьюдента перебрать все пары. Определить поправку Бонферрони. Сделать выводы."""
    
    df = pd.read_csv('4kurs\TIABD\insurance.csv')

    ###поисак уникальных значений region
    mass_region_original=[]
    for i in range(len(df['region'])):
        if ((not mass_region_original)==True) or ((df['region'][i] in mass_region_original)==False):
            mass_region_original.append(df['region'][i])
            pass
    #print('Уникальных регионов ==> ',mass_region_original)
    
    region_chois=df['region']
    bmi_chois= df['bmi']

    chois_frame= pd.DataFrame({'region':region_chois, 'bmi':bmi_chois})
    groups= chois_frame.groupby('region').groups

    print(chois_frame)
    print(chois_frame.head())
    

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

    southwest=bmi_chois[groups['southwest']]
    southeast=bmi_chois[groups['southeast']]
    northwest=bmi_chois[groups['northwest']]
    northeast=bmi_chois[groups['northeast']]

    print('----------------------------------------------------')
    print('southwest\n') 
    print(scipy.stats.ttest_ind(southwest,southeast))
    print(scipy.stats.ttest_ind(southwest,northeast))
    print(scipy.stats.ttest_ind(southwest,northwest))
    print('southeast\n') 
    print(scipy.stats.ttest_ind(southeast,northwest))
    print(scipy.stats.ttest_ind(southeast,northeast))
    print('northwest\n')
    print(scipy.stats.ttest_ind(northwest,northeast))

        #oo,tt=scipy.stats.ttest_ind(bmi_chois[groups[i]],bmi_chois[groups[j]])
        #p-уровень превышает 0.05, следовательно, дисперсии выборок примерно одинаковы  
        #p-значение намного ниже 0.05, следовательно нулевая гипотеза отвергается  (различны)


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

    
    print('Метод Бонферрони ====>', 0.05/6)

    #### Делаем пост-хок тесты 
    tukey = pairwise_tukeyhsd(endog=bmi_chois, groups=region_chois, alpha=0.05)

    tukey.plot_simultaneous()
    ##вопрос , почему мы берем именно эти значения x and y 
    plt.vlines(x=30.5 , ymin=-0.5, ymax=4.5, color= 'red')
    tukey.summary()
    plt.show()


    """
    Видим, что доверительные интервалы white-hispanic и white-asian перекрываются, 
    поэтому пост-хок тесты показали что различия между ними не существенные"""
   

def three():
    
    df = pd.read_csv('4kurs\TIABD\insurance.csv')

    
    sex_chois=df['sex']
    smoker_chois=df['smoker']
    region_chois=df['region']
    bmi_chois= df['bmi']

    chois_frame= pd.DataFrame({'region':region_chois, 'bmi':bmi_chois, 'sex':sex_chois,'smoker':smoker_chois})
    #groups= chois_frame.groupby('region').groups
    #groups= chois_frame.groupby('sex').groups
    
    print(chois_frame.head())
    

    #print('Метод Бонферрони ====>', 0.05/4)
    
    #Двухфакторный дисперсионный анализ с повторениями

    df['combination'] = df.sex + ' / ' + df.region
    model_last = ols('bmi ~ C(region) + C(sex) + C(region):C(sex)', data=chois_frame).fit()

    print('Двухфакторный дисперсионный анализ с повторениями =========> \n',sm.stats.anova_lm(model_last, typ=2))

    tukey2 = pairwise_tukeyhsd(endog=bmi_chois, groups=df.combination, alpha=0.05)
    tukey2.plot_simultaneous()

    ##вопрос , почему мы берем именно эти значения x and y 
    plt.vlines(x=30.5 , ymin=-0.5, ymax=4.5, color= 'red')
    tukey2.summary()
    plt.show()

    




one()
two()
three()



