
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans  
from sklearn.metrics import precision_recall_fscore_support as score
import timeit
from sklearn.metrics import silhouette_score



df =pd.read_csv('4kurs\\TIABD\\Clustering\\cod.csv')

df = df.drop(['name'], axis = 1) # убираем имена так как они нам не нужны

print(df)




x = df.iloc[:,[3,2]].values #Выбор функций Kill-Death и Losses для кластеризации Kmeans


# standardizing the data Стандартизируем знаяения так как они очень сильно различаются в диапозоне 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(x)
#print('DataFrame>>>>>>>>>>\n',pd.DataFrame(data_scaled).describe(),'\n',pd.DataFrame(data_scaled)) 

new_df_col=pd.DataFrame(data_scaled)
###будем выполнять кластеризацию K-средних, поскольку нам нужно классифицировать уровень знаний игроков 
# в Call of Duty. Здесь функция «Соотношение убийств и смертей» является наиболее важным аспектом понимания 
# опыта игроков в игре. Чем выше коэффициент, тем выше опыт игрока. Наряду с этим будет включена еще одна функция,
#  основанная на сильно коррелированной функции для Kill — Death Ratio — «Kill-Sreak», которая также повлияет на опыт игрока

models=[]
score=[]
score2=[]

for i in range(1,11):
    model =KMeans(n_clusters=i, random_state=123, init = 'k-means++').fit(data_scaled)
    models.append(model)
    score.append(model.inertia_)
    score2.append(model.inertia_)


plt.grid()
plt.plot(np.arange(1,11),score,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


###################################### метод локтя ##################################################   
from kneed import KneeLocator

K_value = KneeLocator(range(1,11), score, curve='convex', direction ='decreasing')
print('«правило локтя» =====>',K_value.elbow)  #### метод локтя 
K_value.plot_knee()


start_kmeans = timeit.default_timer()

kmeans = KMeans(n_clusters = 3 ,init = 'k-means++')
y_pred = kmeans.fit_predict(data_scaled)
y_pred_fit=kmeans.fit(data_scaled)
labels=kmeans.fit(data_scaled).labels_

end_kmeans= timeit.default_timer()
time_kmeans=str(end_kmeans-start_kmeans)
print('Время kmeans>>>>>>>>>>>>>>>>>> ',time_kmeans)
from sklearn.metrics import silhouette_score
print('Коэффициента силуэта достигает максимума при=====>',silhouette_score(x,y_pred))







# plt.figure(figsize=(10,5))
# plt.scatter(x[y_pred == 0,0], x[y_pred == 0,1], s=100, c='blue', label = 'Cluster1')
# plt.scatter(x[y_pred == 1,0], x[y_pred == 1,1], s=100, c='red', label = 'Cluster2')
# plt.scatter(x[y_pred == 2,0], x[y_pred == 2,1], s=100, c='green', label = 'Cluster3')
# plt.title('k-Means алгоритм')
# plt.xlabel('Kill Streaks')
# plt.ylabel('Kill-Death ratio')



from sklearn.cluster import AgglomerativeClustering
##################################################### алгоритма иерархической кластеризации ######################

start_alg_clust = timeit.default_timer()

aggl_clust_model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_model_aggl=aggl_clust_model.fit_predict(data_scaled)


end_alg_clust= timeit.default_timer()
time_alg_clust=str(end_alg_clust-start_alg_clust)
print('Время алгоритма иерархической кластеризации>>>>>>>>>>>>>>>>>> ',time_alg_clust)

plt.figure(figsize=(10,5))
plt.scatter(x[y_model_aggl == 0,0], x[y_model_aggl == 0,1], s=100, c='blue', label = 'Cluster1')
plt.scatter(x[y_model_aggl == 1,0], x[y_model_aggl == 1,1], s=100, c='red', label = 'Cluster2')
plt.scatter(x[y_model_aggl == 2,0], x[y_model_aggl == 2,1], s=100, c='green', label = 'Cluster3')
plt.title('алгоритма иерархической кластеризации ')
plt.xlabel('Kill Streaks')
plt.ylabel('Kill-Death ratio')

plt.show()

##################################################### алгоритма DBSCAN  ########################################33######################

from sklearn.cluster import DBSCAN
start_dbscan = timeit.default_timer()
clustering = DBSCAN(eps=12, min_samples=5).fit(data_scaled)
print(clustering)
algor_dbscan=clustering.labels_


end_dbscan= timeit.default_timer()
time_dbscan=str(end_dbscan-start_dbscan)
print('Время алгоритма иерархической кластеризации>>>>>>>>>>>>>>>>>> ',time_dbscan)
print('algor_dbscan>>>>>>>>>>>>>>>>>>',algor_dbscan)


#plt.show()

from sklearn.manifold import TSNE

embed = TSNE(
    n_components=2, # значение по умолчанию=2. Размерность вложенного пространства.
    perplexity=20, # значение по умолчанию=30.0. Перплексия связана с количеством ближайших соседей, которое используется в других алгоритмах обучения на множествах.
    early_exaggeration=12, # значение по умолчанию=12.0. Определяет, насколько плотными будут естественные кластеры исходного пространстве во вложенном пространстве и сколько места будет между ними. 
    learning_rate=200, # значение по умолчанию=200.0. Скорость обучения для t-SNE обычно находится в диапазоне [10.0, 1000.0]. Если скорость обучения слишком высока, данные могут выглядеть как "шар", в котором любая точка приблизительно равноудалена от ближайших соседей. Если скорость обучения слишком низкая, большинство точек могут быть похожими на сжатое плотное облако с незначительным количеством разбросов. 
    n_iter=5000, # значение по умолчанию=1000. Максимальное количество итераций для оптимизации. Должно быть не менее 250.
    n_iter_without_progress=300, # значение по умолчанию=300. Максимальное количество итераций без прогресса перед прекращением оптимизации, используется после 250 начальных итераций с ранним преувеличением.
    min_grad_norm=0.0000001, # значение по умолчанию=1e-7. Если норма градиента ниже этого порога, оптимизация будет остановлена.
    metric='euclidean', # значение по умолчанию='euclidean', Метрика, используемая при расчете расстояния между экземплярами в массиве признаков.
    init='random',# {'random', 'pca'} или ndarray формы (n_samples, n_components), значение по умолчанию='random'. Инициализация вложения.
    verbose=0, # значение по умолчанию=0. Уровень детализации.
    random_state=42, # экземпляр RandomState или None, по умолчанию=None. Определяет генератор случайных чисел. Передача int для воспроизводимых результатов при многократном вызове функции.
    method='barnes_hut', # значение по умолчанию='barnes_hut'. По умолчанию алгоритм вычисления градиента использует аппроксимацию Барнса-Хата, работающую в течение времени O(NlogN). метод='exact' будет работать по более медленному, но точному алгоритму за время O(N^2). Следует использовать точный алгоритм, когда количество ошибок ближайших соседей должно быть ниже 3%.
    angle=0.5, # значение по умолчанию=0.5. Используется только если метод='barnes_hut' Это компромисс между скоростью и точностью в случае T-SNE с применением алгоритма Барнса-Хата.
    n_jobs=-1, # значение по умолчанию=None. Количество параллельных заданий для поиска соседей. -1 означает использование всех процессоров.
        )

new_df_col['labels']=labels

X_embedded = embed.fit_transform(new_df_col)
fig = px.scatter(None, x=X_embedded[:,0], y=X_embedded[:,1], 
                    labels={
                        "x": "Kill Streaks",
                        "y": "Kill-Death ratio",
                    },
                    opacity=1,color=new_df_col['labels'])

    # Изменение цвета фона графика
fig.update_layout(dict(plot_bgcolor = 'white'))

    # Обновление линий осей
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                    zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                    showline=True, linewidth=1, linecolor='black')

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                    zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                    showline=True, linewidth=1, linecolor='black')

    # Установка названия рисунка
fig.update_layout(title_text="t-SNE")

    # Обновление размера маркера
fig.update_traces(marker=dict(size=10))

fig.show()
    


print('_______________________________________________\n')
print('алгоритм                   | Время выполнения         \n')
print('_______________________________________________\n')
print('метод локтя                |',time_kmeans,'\n')
print('_______________________________________________\n')
print('иерархической кластеризации|',time_alg_clust,'\n')
print('_______________________________________________\n')
print('DBSCAN                     |',time_dbscan,'\n')
print('_______________________________________________\n')

##### нужно доделать последний алгоритм и вывести таблицу времени алгоритмов 
