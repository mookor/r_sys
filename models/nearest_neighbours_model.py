import numpy as np
import scipy.sparse as sparse
from utils.prepare_data import Data_worker
from tqdm import tqdm
import pandas as pd
import implicit
import pickle


class Knn:
    def __init__(self, data, K=1):
        """
        Модель на основе алгоритма ближайших соседей

        K - количество соседей
        data - model_df
        """
        self.K = K
        self.data = data
        self.create_sprace_arrays()

    def create_sprace_arrays(self):
        """
        Создание разряженных матриц
        """
        self.sparse_item_user = sparse.csr_matrix(
            (
                self.data["count"].astype(float),
                (self.data["product"], self.data["user"]),
            )
        )
        self.sparse_user_item = sparse.csr_matrix(
            (
                self.data["count"].astype(float),
                (self.data["user"], self.data["product"]),
            )
        )
        print(self.sparse_item_user.shape)

    def fit(self):
        """
        Обучение модели
        """
        self.model = implicit.nearest_neighbours.CosineRecommender(K=self.K)
        self.model.fit(self.sparse_item_user)

    def predict_all(self, uniq_users, encode_products, encode_users, n=10):
        """
        предсказать продукты для всех user в массиве uniq_users
        n - количество предлагаемых продуктов
        """

        users_dict = {}
        for user in tqdm(uniq_users):
            decode_user = encode_users[encode_users["user_id"] == user]["user"].iloc[0]
            recommended = self.model.recommend(
                decode_user,
                self.sparse_user_item,
                filter_already_liked_items=False,
                recalculate_user=True,
                N=n,
            )
            user_rec_items = []
            for item in recommended:
                idx, score = item
                rec_item = (
                    encode_products["product_id"]
                    .loc[encode_products["product"] == idx]
                    .iloc[0]
                )
                user_rec_items.append(rec_item)
            users_dict[user] = user_rec_items
        return users_dict

    def predict_for_user(self, user, encode_products, encode_users, n=10):
        """
        предсказать продукты для конкретного пользователя
        n - количество предлагаемых продуктов
        """
        users_dict = {}
        decode_user = encode_users[encode_users["user_id"] == user]["user"].iloc[0]
        recommended = self.model.recommend(
            decode_user,
            self.sparse_user_item,
            N=n,
            filter_already_liked_items=False,
            recalculate_user=True,
        )
        user_rec_items = []
        for item in recommended:
            idx, score = item
            rec_item = (
                encode_products["product_id"]
                .loc[encode_products["product"] == idx]
                .iloc[0]
            )
            user_rec_items.append(rec_item)
        users_dict[user] = user_rec_items
        return users_dict

    def create_submission(
        self, users_dict, uniq_users, path="results/nearest_neighbours25.csv"
    ):
        """
        Создание структуры для отправки на Kaggle
        """
        for user in uniq_users:
            if user in users_dict.keys():
                users_dict[user] = " ".join(str(x) for x in users_dict[user])
        df_sub = pd.DataFrame(users_dict.items(), columns=["user_id", "product_id"])
        df_sub.to_csv(path, index=False)

    def save_model(self, weights_path):
        self.model.save(weights_path)

    def load_model(self, weights_path):
        self.model = implicit.nearest_neighbours.CosineRecommender(K=self.K)
        self.model.load(weights_path)


if __name__ == "__main__":
    d = Data_worker()

    d.read_counted_df()
    d.modef_df_to_csv()
    d.read_model_df()
    d.read_products_encode_df()
    d.read_user_encode_df()
    d.read_uniq_users()
    data = d.model_df
    k = Knn(data, K=1)
    k.fit()
    k.save_model(weights_path="weights/knn_model")
    k.load_model(weights_path="weights/knn_model")
    # asd = k.predict_all(d.uniq_users, d.encode_products, d.encode_users)
    # k.create_submission(asd, d.uniq_users)
