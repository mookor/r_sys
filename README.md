# r_sys
## Оглавление
[Подготовка окружения](#Подготовка_окружения)  
>[Инструкция к использованию](#Инструкция_использованию)  
>> [Загрузка данных](#data)  
>>            [Knn_model](#knn)    
>>            [Als_model](#als)    
>>            [Top10_model](#top10)    

[Основной пайплайн](#Пайплайн)  
[Результат](#result)  
[Быстырй старт](#fast_start)  
## Подготовка окружения <a name="Подготовка_окружения"></a> 
1) создать виртуальное окружение на основе python3.6
```virtualenv venv --python=python3.6```
2) активировать окружение
```source venv/bin/activate```
3) установить нужные пакеты 
```pip install -r requirements.txt```

## Инструкция к использованию <a name="Инструкция_использованию"></a>
### Загрузка данных<a name="data"></a>
для подготовки данных их сырых таблиц transactions и products необходимо выполнить  
```prepare_all_data(Data_worker())```
После выполнения получаем следующие файлы
```
encode_products.csv   
model_df.csv  
transactions.csv  
user_product_counted.csv encode_users.csv  
products.csv  
uniq_users.npy
```
В случае, если данные подготовлены - следует использовать  
```read_for_model_fit(Data_worker())```  для загрузки нужных данных  
### Knn <a name="knn"></a>
для использования модели на основе алгоритма Ближайших соседей
```
#  Knn - Model
########################################################
k = Knn(data, K=1)  # инициализация knn модели
k.fit()  # обучение
k.save_model(weights_path="weights/knn_model")  # cохранение весов

predict_user = k.predict_for_user(
    user=1,
    encode_products=data_worker.encode_products,
    encode_users=data_worker.encode_users,
    n=3,
)  # предсказание для конкретного пользователя , количество предлагаемых продуктов - n
print(predict_user)

predict = k.predict_all(
    data_worker.uniq_users, data_worker.encode_products, data_worker.encode_users
)  # предсказание для всех пользователей , информация о которых есть в таблице Transactions
k.create_submission(predict, data_worker.uniq_users) # cоздание файла для kaggle
#############################################################
```
### Als<a name="als"></a>
Для использования модели Als 
```
#  ALS - Model
########################################################
als_model = Als(data, factors=15, iterations=1)  # инициализация als модели
als_model.fit()  # обучение
als_model.save_model(weights_path="weights/knn_model")  # cохранение весов

predict_user = als_model.predict_for_user(
    user=1,
    encode_products=data_worker.encode_products,
    encode_users=data_worker.encode_users,
    n=3,
)  # предсказание для конкретного пользователя , количество предлагаемых продуктов - n
print(predict_user)

predict = als_model.predict_all(
    data_worker.uniq_users, data_worker.encode_products, data_worker.encode_users
)  # предсказание для всех пользователей , информация о которых есть в таблице Transactions
als_model.create_submission(predict, data_worker.uniq_users) # cоздание файла для kaggle
##################################################################################
```
### Top_10<a name="top10"></a>
Так же можно проверитькак работаю рекомендации на основе топ10 по полуряности продуктов для каждого юзера
```
python -m models.most_popular_products
```
### Пайплайн <a name="Пайплайн"></a>
Пример полного пайплайна можно посмотреть [здесь](https://github.com/mookor/r_sys/blob/main/main.py)  
запускать нужно следующим образом
```
python -m main
```

## Результаты<a name="result"></a>
| model        | private           | public  |
| ------------- |:-------------:| -----:|
| Knn(k=10)     | 0.23053 | 0.23300 |
| Knn(k=1)      | 0.23091      |   0.23292 |
| Als(factors = 30 , iter = 8 ) | 0.13253     |   0.13161 |
| Top10_orders | 0.27648 |   0.27815 |

Как видно , самую лучшую метрику показали топ10_популярных для каждого пользователя , но вряд ли такие рекомендации можно считать действительно полезными , ибо ничего нового не рекомендуется   
Другие модели показали метрику ниже, но сейчас модель действительно рекомендует товары

Модель можно попробовать улучшить - учитывать дополнительную информацию по каждой модели , например засунуть все получившееся в катбуст и посмотреть что будет

## Быстрый старт<a name="fast_start"></a>
чтобы не ждать, пока данные обрабодаются - можно скачать все подготовленные таблицы [здесь](https://disk.yandex.ru/d/T1vBTMqG5e_2gg)  
После скачивания - разархивировать в папку data
