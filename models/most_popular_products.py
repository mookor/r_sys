from utils.prepare_data import Data_worker
import pandas as pd
from tqdm import tqdm


def calc_most_popular_per_user(transactions, uniq_users):
    counter_all_users = {}
    for user in tqdm(uniq_users):
        user_transactions = transactions[transactions["user_id"] == user]
        list_of_orders = user_transactions["product_id"]
        order_counter = list_of_orders.value_counts()
        order_counter_dict = order_counter.to_dict()
        sorted_counter_list = [k for k in order_counter_dict.keys()][:10]
        counter_all_users[user] = sorted_counter_list
    return counter_all_users


def create_submissions(counter_all_users, uniq_users, path="results/top10.csv"):
    for user in uniq_users:
        if user in counter_all_users.keys():
            counter_all_users[user] = " ".join(str(x) for x in counter_all_users[user])
    df = pd.DataFrame(counter_all_users.items(), columns=["user_id", "product_id"])
    df.to_csv(path, index=False)

if __name__ == "__main__":
    worker = Data_worker()
    worker.read_data()
    worker.read_uniq_users()
    counter_all_users = calc_most_popular_per_user(worker.transactions, worker.uniq_users)
    create_submissions(counter_all_users, worker.uniq_users)
