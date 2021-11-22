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
read_for_model_fit(data_worker)

data = data_worker.model_df

k = Knn(data, K=1)
k.fit()
k.save_model(weights_path = "weights/knn_model")
predict_user = k.predict_for_user(1, data_worker.encode_products, data_worker.encode_users , n= 3)
# predict = k.predict_all(data_worker.uniq_users, data_worker.encode_products, data_worker.encode_users)
k.create_submission(predict_user, data_worker.uniq_users)