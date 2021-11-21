import pandas as pd
import numpy as np


def make_encode_df(model_df, columns, path):
    encode_df = model_df[columns]
    encode_df = encode_df.drop_duplicates()
    encode_df.to_csv(path, index=False)


def make_df_for_model(counted_df, model_df_path):
    df = counted_df.copy()
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
    counter_all_users = {}
    for user in uniq_users:
        user_transactions = transactions[transactions["user_id"] == user]
        list_of_orders = user_transactions["product_id"]
        order_counter = list_of_orders.value_counts()
        order_counter_dict = order_counter.to_dict()
        counter_all_users[user] = order_counter_dict
    counted_arr = []
    for user in counter_all_users.keys():
        for prod, cnt in counter_all_users[user].items():
            counted_arr.append([user, prod, cnt])
    make_counted_df = pd.DataFrame(
        counted_arr, columns=["user_id", "product_id", "count"]
    )
    make_counted_df.to_csv(counted_df_path, index=False)
