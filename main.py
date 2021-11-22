from models.als import Als
from models.nearest_neighbours_model import Knn
from utils.prepare_data import Data_worker


def prepare_all_data(data_worker):
    data_worker.read_data()
    data_worker.uniq_users_to_numpy()
    data_worker.read_uniq_users()
    data_worker.counted_df_to_csv()
    data_worker.read_counted_df()
    data_worker.modef_df_to_csv()
    data_worker.read_model_df()
    data_worker.make_products_encode()
    data_worker.make_users_encode()
    data_worker.read_user_encode_df()
    data_worker.read_products_encode_df()


def read_all(data_worker):
    data_worker.read_data()
    data_worker.read_uniq_users()
    data_worker.read_counted_df()
    data_worker.read_model_df()
    data_worker.read_user_encode_df()
    data_worker.read_products_encode_df()


def read_for_model_fit(data_worker):
    data_worker.read_model_df()
    data_worker.read_uniq_users()
    data_worker.read_user_encode_df()
    data_worker.read_products_encode_df()


data_worker = Data_worker()
# в случае , если все данные подготовлены  - следует использовать метод read_for_model_fit() для загрузки данных перед обучением
# в случае, если данные не подготовлены и есть только сырые таблицы transactions и products - использовать метод prepare_all_data()
# prepare_all_data(data_worker)
read_for_model_fit(data_worker)


data = data_worker.model_df
#  KNN - Model
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
##################################################################################


#  ALS - Model
########################################################
k = Als(data, factors=15, iterations=1)  # инициализация als модели
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
##################################################################################
