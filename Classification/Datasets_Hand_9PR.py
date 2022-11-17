
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np  
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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

sns.countplot(x=64,data=new_df)# выводим гистрограмму 
#plt.show()


x_train , x_test, y_train , y_test = train_test_split(predictors, target, train_size=0.9, random_state=271,
 shuffle=True # перемешимает знач 
        )

print(
    'Размер для x_train ', x_train.shape,'\n',
    'Размер для x_test ', x_test.shape,'\n',
    'Размер для y_train ', y_train.shape,'\n',
    'Размер для y_test ', y_test.shape,'\n',
)


#####################################################################################################################################################################

start = timeit.default_timer()

model = LogisticRegression(random_state=271)
model.fit(x_train,y_train)

y_pred_logist = model.predict(x_test) # тесты на основе обученной модели 
    
# print('Предсазанные значения \n',len(y_predict),y_predict[:15])
# print('Исходные значения \n', np.array(y_test[:15])  )

fig = px.imshow(confusion_matrix(y_test,y_pred_logist),text_auto=True)
fig.update_layout( xaxis_title='Target',yaxis_title='Prediction')

#fig.show()Logist regres

end = timeit.default_timer()
time_logist=str(end-start)


print('логистическая регрессия=============\n ',classification_report(y_test, y_pred_logist))
print('Время выполнения алгоритма логистической регрессии===>',time_logist)
print('####################################################################################')

#####################################################################################################################################################################
"""
        Жесткая маржа:стремится найти лучшую гиперплоскость, не допуская ни одной формы неправильной классификации.
        Мягкая маржа:мы добавляем степень терпимости в SVM. Таким образом, мы позволяем модели произвольно ошибочно 
        классифицировать несколько точек данных, если это может привести к идентификации гиперплоскости, способной лучше обобщать невидимые данные."""

start = timeit.default_timer()

svc= SVC(kernel='rbf',C=15)
svc.fit(x_train,y_train)

y_pred_svc=svc.predict(x_test)

# print('Предсазанные значения \n',len(y_pred_svc),y_pred_svc[:15])
# print('Исходные значения \n', np.array(y_test[:15])  )

fig = px.imshow(confusion_matrix(y_test,y_pred_svc),text_auto=True)
fig.update_layout( xaxis_title='Target',yaxis_title='Prediction')

#fig.show() график svc
    
end = timeit.default_timer()
time_svc=str(end-start)
print('SVC=============\n ',classification_report(y_test, y_pred_svc))
print('Время выполнения алгоритма SVC==>',time_svc)
print('####################################################################################')

#####################################################################################################################################################################

start = timeit.default_timer()

number = np.arange(3,10,25)
model_knn=KNeighborsClassifier()
params={'n_neighbors':number}


grid_search= GridSearchCV(estimator=model_knn, param_grid=params)
grid_search.fit(x_train,y_train)

print(grid_search.best_score_)
print(grid_search.best_estimator_)

y_pred_knn=grid_search.predict(x_test)

end = timeit.default_timer()
time_knn=str(end-start)

# print('Предсазанные значения \n',len(y_pred_knn),y_pred_knn[:15])
# print('Исходные значения \n', np.array(y_test[:15])  )

print('KNN=============\n ',classification_report(y_pred_knn,y_test))
print('Время выполнения алгоритма KNN===>',time_knn)

fig = px.imshow(confusion_matrix(y_test,y_pred_knn),text_auto=True)
fig.update_layout( xaxis_title='Target',yaxis_title='Prediction')

#fig.show() # график KNN
    

################################# последнее задание #################################
print('\n##################Последнее задание ######################\n')
metrics = []
models = ['Логистическая регрессия', 'SVC', 'KNN']
predictions=[y_pred_logist, y_pred_svc, y_pred_knn]

for lab,i in zip(models, predictions):
    precision, recall, fscore, _ = score(y_test, i, average='weighted')
    accuracy = accuracy_score(y_test, i)
    auc = roc_auc_score(label_binarize(y_test, classes=[0,1,2,3]),
                        label_binarize(i, classes=[0,1,2,3]),
                        average='weighted')
    metrics.append(pd.Series({'precision':precision, 'recall':recall,
                              'fscore':fscore,
                              'accuracy':accuracy,}, name=lab))

metrics = pd.concat(metrics, axis=1)

print(metrics)
print('\n##################Время выполнения алгоритмов######################\n')
print('\n','Время Логистической регресии',time_logist,'\n',
      'Время SVC',time_svc,'\n',
      'Время KNN',time_knn,'\n')




# param_kernel=('linear','rbf', 'poly', 'sigmoid')
# parameters1 = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# print('1')
# parameters= {'kernel':param_kernel}
# print('1')
# model = SVC()
# print('1')
# grid_search_svm= GridSearchCV(estimator=model, param_grid=parameters1)
# print('1')
# grid_search_svm.fit(x_train,y_train)
# print('1')

# best_model=grid_search_svm.best_estimator_
# print('1')
# print(best_model.kernel)