
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np  
import timeit
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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import time
import catboost as cb


"""Задача классификации – это задача обучения с учителем, исходные данные размечены, то есть известны классы объектов. 
Основная цель классификации – это определение класса, к которому относится некоторый объект."""

df_0 =pd.read_csv('4kurs\\TIABD\\testDatasets\\0.csv', header=None ) # камень - 0, ножницы - 1, бумага - 2, хорошо - 3

df_1 =pd.read_csv('4kurs\\TIABD\\testDatasets\\1.csv',header=None ) # камень - 0, ножницы - 1, бумага - 2, хорошо - 3 

df_2 =pd.read_csv('4kurs\\TIABD\\testDatasets\\2.csv',header=None ) #камень - 0, ножницы - 1, бумага - 2, хорошо - 3

df_3 =pd.read_csv('4kurs\\TIABD\\testDatasets\\3.csv',header=None ) # камень - 0, ножницы - 1, бумага - 2, хорошо - 3

# print('\n',df_0.head(),'zero')
# print('\n',df_1.head(),'one')
# print('\n',df_2.head(),'two')
# print('\n',df_3,'three')

new_df = pd.concat([df_0,df_1,df_2,df_3]) # соединяем все dataframe
predictors= new_df.loc[:,1:63] # делаем срез для того чтобы нен входило значение последнего столбца 
target= new_df[64] # наши значения от 0 до 3



x_train , x_test, y_train , y_test = train_test_split(predictors, target, train_size=0.9, random_state=271,
 shuffle=True # перемешимает знач 
        )

print(
    'Размер для x_train ', x_train.shape,'\n',
    'Размер для x_test ', x_test.shape,'\n',
    'Размер для y_train ', y_train.shape,'\n',
    'Размер для y_test ', y_test.shape,'\n',
)



def f1():

    clf=RandomForestClassifier(max_depth=17, min_samples_split=10).fit(x_train,y_train)
    clf_predict_train=clf.predict(x_train)
    print('F1 мера для тренировочных данных\n', f1_score(clf_predict_train, y_train, average='macro'))

    clf_predict_test=clf.predict(x_test)
    print('F2 мера для тестовых данных\n', f1_score(clf_predict_test, y_test, average='macro'))
    pass
f1()
def f2():
        random_forest = RandomForestClassifier()
        params_grid = {
            'max_depth': [12, 18],
            'min_samples_leaf': [3, 10],
            'min_samples_split': [6, 12]
        }
        start_time = time.time()
        grid_search_random_forest = GridSearchCV(estimator=random_forest,
                                                param_grid=params_grid,
                                                scoring='f1_macro',
                                                cv=4)
        print(grid_search_random_forest.fit(x_train, y_train))

        best_model = grid_search_random_forest.best_estimator_
        print(best_model)
        time1 = time.time() - start_time

        y_preds_d = best_model.predict(x_train)
        print('\n')
        print('F1 мера для тренировочных данных', f1_score(
        y_preds_d, y_train, average='macro'))

        y_preds = best_model.predict(x_test)
        score1 = f1_score(y_preds, y_test, average='macro')
        print('F1 мера для тестовых данных', score1)

        def f3():
            start_time = time.time()
            model_catboost_clf = cb.CatBoostClassifier(iterations=1000)
            model_catboost_clf.fit(x_train, y_train)
            time2 = time.time() - start_time

            y_preds_t = model_catboost_clf.predict(x_train)
            print('F1 мера для тренировочных данных', f1_score(
                y_preds_t, y_train, average='macro'))

            y_preds = model_catboost_clf.predict(x_test)
            score2 = f1_score(y_preds, y_test, average='macro')
            print('F1 мера для тестовых данных', score2)

            def f4():
                df = pd.DataFrame({
                    'Algorithm': ['Баггинг', 'Бустинг'],
                    'time': [time1, time2],
                    'Efficiency': [score1, score2]
                })
                print(df)
            f4()
        f3()
f2()