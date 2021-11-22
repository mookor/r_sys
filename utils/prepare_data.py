import sys

sys.path.append("utils")
from . import data_utils
import os
import pandas as pd
import numpy as np


class Data_worker:
    def __init__(
        self,
        products_path="data/products.csv",
        transactions_path="data/transactions.csv",
        uniq_users_path="data/uniq_users.npy",
        counted_df_path="data/user_product_counted.csv",
        model_df_path="data/model_df.csv",
        encode_products_path="data/encode_products.csv",
        encode_users_path="data/encode_users.csv",
    ):
        self.products_path = products_path
        self.transactions_path = transactions_path
        self.uniq_users_path = uniq_users_path
        self.counted_df_path = counted_df_path
        self.encode_products_path = encode_products_path
        self.encode_users_path = encode_users_path
        self.model_df_path = model_df_path
        self.products = None
        self.transactions = None
        self.uniq_users = None
        self.counted_df = None
        self.model_df = None
        self.encode_users = None
        self.encode_products = None

    def read_data(self):
        """
        Считывание transactions и products
        """
        self.products, self.transactions = data_utils.read_data(
            self.products_path, self.transactions_path
        )
        print("Успешно считаны основные данные")

    def uniq_users_to_numpy(self):
        """
        Функция для выделения и сохранения уникальных пользователей
        """
        if self.transactions is None:
            raise ValueError("Данные не загружены")
        users = self.transactions["user_id"]
        data_utils.save_uniq_users(users, self.uniq_users_path)
        print("Уникальные пользователи сохранены")

    def modef_df_to_csv(self):
        """
        Создание и сохранение датасета для обучения
        """
        if self.counted_df is None:
            raise ValueError("Данные не загружены")
        data_utils.make_df_for_model(self.counted_df, self.model_df_path)
        print("Набор данных для обучения модели создан успешно")

    def counted_df_to_csv(self):
        """
        Создание и сохранение набора данных с подсчитанными заказами
        """
        if self.transactions is None:
            raise ValueError("Данные не загружены")
        if self.uniq_users is None:
            raise ValueError("Данные не загружены")
        data_utils.make_counted_data(
            self.transactions, self.uniq_users, self.counted_df_path
        )
        print("Подсчитанный набор данных создан успешно")

    def make_products_encode(self):
        """
        Создание и сохранение энкодера для продуктов
        """
        if self.model_df is None:
            raise ValueError("Данные не загружены")
        data_utils.make_encode_df(
            self.model_df, ["product_id", "product"], self.encode_products_path
        )
        print("Данные для декодирования продуктов созданы успешно")

    def make_users_encode(self):
        """
        Создание и сохранение энкодера для пользователей
        """
        if self.model_df is None:
            raise ValueError("Данные не загружены")
        data_utils.make_encode_df(
            self.model_df, ["user_id", "user"], self.encode_users_path
        )
        print("Данные для декодирования пользователей созданы успешно")

    def read_counted_df(self):
        """
        Считывание датасета с подсчитанными заказами
        """
        if not os.path.exists(self.counted_df_path):
            raise OSError(
                "Неправильный путь к датасету с подсчитанными данными / файл не существует"
            )
        self.counted_df = pd.read_csv(self.counted_df_path)
        print("Успешно считаны подсчитанные данные")

    def read_uniq_users(self):
        """
        Считывание файла со списком уникальных пользователей
        """
        if not os.path.exists(self.uniq_users_path):
            raise OSError(
                "Неправильный путь к уникальным пользователям / файл не существует"
            )
        self.uniq_users = np.load(self.uniq_users_path)
        print("Уникальные пользователи считаны успешно")

    def read_model_df(self):
        """
        Считывание датасета для обучения модели
        """
        if not os.path.exists(self.model_df_path):
            raise OSError(
                "Неправильный путь к датасету для обучения / файл не существует"
            )
        self.model_df = pd.read_csv(self.model_df_path)
        print("Успешно считаны данные для обучения модели")

    def read_user_encode_df(self):
        """
        Считывание энкодера для пользователей
        """
        if not os.path.exists(self.encode_users_path):
            raise OSError(
                "Неправильный путь к енкодеру пользователей / файл не существует"
            )
        self.encode_users = pd.read_csv(self.encode_users_path)
        print("Успешно считаны данные для декодирования пользователей")

    def read_products_encode_df(self):
        """
        Считывание энкодера для продуктов
        """
        if not os.path.exists(self.encode_products_path):
            raise OSError("Неправильный путь к енкодеру продуктов / файл не существует")
        self.encode_products = pd.read_csv(self.encode_products_path)
        print("Успешно считаны данные для декодирования продуктов")


if __name__ == "__main__":
    d = Data_worker()
    d.read_data()
    d.uniq_users_to_numpy()
    d.read_uniq_users()
    d.counted_df_to_csv()
    d.read_counted_df()
    d.modef_df_to_csv()
    d.read_model_df()
    d.make_products_encode()
    d.make_users_encode()
    d.read_user_encode_df()
    d.read_products_encode_df()
