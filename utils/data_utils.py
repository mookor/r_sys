import pandas as pd
import numpy as np
from tqdm import tqdm


def make_encode_df(model_df, columns, path):
    encode_df = model_df[columns]
    encode_df = encode_df.drop_duplicates()
    encode_df.to_csv(path, index=False)


def make_df_for_model(counted_df, model_df_path):
    df = counted_df.copy()
    # конвртирование данных в коды категорий
    # данный формат необходим для обучения моделей , основанных на impicit
    df["user_id"] = df["user_id"].astype("category")
    df["product_id"] = df["product_id"].astype("category")
    df["user"] = df["user_id"].cat.codes
    df["product"] = df["product_id"].cat.codes

    df["count"] = df["count"].astype("category")
    df["count"] = df["count"].cat.codes
    df.to_csv(model_df_path, index=False)


def save_uniq_users(user_arr, uniq_users_path):
    uniq_users = user_arr.unique()
    np.save(uniq_users_path, uniq_users)


def read_data(products_path, transactions_path):
    """
    return products , transactions
    """
    products = pd.read_csv(products_path)
    transactions = pd.read_csv(transactions_path)
    return products, transactions


def make_counted_data(transactions, uniq_users, counted_df_path):
    """
    функция подсчитывания заказов
    """
    counter_all_users = {}
    for user in tqdm(uniq_users):
        user_transactions = transactions[
            transactions["user_id"] == user
        ]  # выбор всех транзакций для определенного пользователя - user
        list_of_orders = user_transactions[
            "product_id"
        ]  # получения списка продуктов из всех транзакций пользователя
        order_counter = (
            list_of_orders.value_counts()
        )  # подсчет кол-ва заказов для каждого продукта
        order_counter_dict = order_counter.to_dict()
        counter_all_users[user] = order_counter_dict
    counted_arr = []

    # разложения словаря в более удобный формат
    for user in counter_all_users.keys():
        for prod, cnt in counter_all_users[user].items():
            counted_arr.append([user, prod, cnt])
    make_counted_df = pd.DataFrame(
        counted_arr, columns=["user_id", "product_id", "count"]
    )

    # сохранение подсчитанного датасета
    make_counted_df.to_csv(counted_df_path, index=False)
