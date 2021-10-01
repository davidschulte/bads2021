from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np

"""This data transformer creates new features and cleans the data."""
class DataTransformer(TransformerMixin):

    def __init__(self, use_delivery_date=True):
        self.use_delivery_date = use_delivery_date
        self.cat_features = ['item_size', 'item_color', 'user_title', 'user_state']
        self.date_features = ['order_date', 'delivery_date', 'user_dob', 'user_reg_date']
        self.id_features = ['item_id', 'brand_id', 'user_id']

    def fit(self, X, y):

        X = self.convert_data_types(X)
        X = self.create_features_independent(X, self.use_delivery_date)

        self.user_age_median = X['user_age'].quantile(0.5)

        if self.use_delivery_date:
            self.delivery_span_median = X['delivery_span'].quantile(0.5)

        df_user_returns = pd.concat([X['user_id'],y], axis=1)

        df_user_returns['return'] = df_user_returns.groupby(['user_id'])['return'].transform('mean').astype(np.float32)

        self.user_avg_returns, self.item_avg_returns = self.get_averages(X,y)

        return self

    def transform(self, X):
        X = self.convert_data_types(X)
        X = self.create_features_independent(X, self.use_delivery_date)
        X = self.create_features_dependent( X)
        X = self.clean(X)
        X = self.drop_columns(X)
        X = self.unskew(X)
        return X


    def convert_data_types(self, data):
        for feature in self.cat_features:
            data[feature] = data[feature].astype('category')
        for feature in self.date_features:
            data[feature] = pd.to_datetime(data[feature])
        for feature in self.id_features:
            data[feature] = data[feature].astype('int32')

        data['item_price'] = data['item_price'].astype('float32')

        return data

    def create_features_independent(self, data, use_delivery_date=True):
        data['user_age'] = (data['order_date'] - data['user_dob']).dt.days / 365

        data['item_is_free'] = data['item_price'] == 0

        if use_delivery_date:
            data['delivery_span'] = (data['delivery_date'] - data['order_date']).dt.days.astype(np.float32)
            data['order_delivered'] = ~data['delivery_date'].isnull()

        return data

    def create_features_dependent(self, data):
        data['order_num_items'] = data.groupby(['user_id', 'order_date'])['user_id'].transform('size').astype(
            np.int32)

        data['item_multiple_orders'] = data.groupby(['user_id', 'order_date', 'item_id'])['user_id'].transform(
            'size').astype(np.int32) > 1

        data['user_orders'] = data.groupby(['user_id'])['user_id'].transform('size').astype(np.float32)

        data['item_popularity'] = data.groupby(['item_id'])['item_id'].transform('size').astype(np.int32)

        data['item_color_popularity'] = data.groupby(['item_color'])['item_color'].transform('size').astype(np.int32)

        data['user_avg_return'] = data['user_id'].apply(self.assign_return_rate, args=(self.user_avg_returns,)).astype(np.float32)

        data['item_avg_return'] = data['item_id'].apply(self.assign_return_rate, args=(self.item_avg_returns,)).astype(np.float32)

        return data

    def clean(self, data):
        data.loc[data['user_age'] < 16, 'user_age'] = np.nan
        data.loc[data['user_age'] > 100, 'user_age'] = np.nan

        data.loc[data['user_age'].isnull(), 'user_age'] = self.user_age_median

        data.loc[data['item_price'] > 400, 'item_price'] = 400
        data.loc[data['order_num_items'] > 40, 'order_num_items'] = 40
        data.loc[data['user_orders'] > 100, 'user_orders'] = 100

        if self.use_delivery_date:
            data.loc[data['delivery_span'] < 0, 'delivery_span'] = np.nan
            data.loc[data['delivery_span'].isnull(), 'delivery_span'] = self.delivery_span_median
            data.loc[data['delivery_span'] > 50, 'delivery_span'] = 50

        return data

    def drop_columns(self, data):
        data = data.drop(self.cat_features + self.date_features + self.id_features, axis=1)

        return data

    def assign_return_rate(self, id, dictionary):
        if id in dictionary:
            return dictionary[id]

        return np.mean(list(dictionary.values()))


    def unskew(self, data):
        skewed = ['item_price', 'order_num_items',  'user_orders',
                  'item_popularity']
        if self.use_delivery_date:
            skewed.append('delivery_span')
        for feature in skewed:
            data[feature] += 1
            data[feature] = data[feature].apply(np.log)

        return data

    def get_averages(self, X, y):
        df = pd.concat([X[['user_id', 'item_id']], y], axis=1)

        df['user_avg_return'] = df.groupby(['user_id'])['return'].transform('mean').astype(np.float32)
        df['item_avg_return'] = df.groupby(['item_id'])['return'].transform('mean').astype(np.float32)

        user_avg_returns = df.set_index('user_id')['user_avg_return'].to_dict()

        item_avg_returns = df.set_index('item_id')['item_avg_return'].to_dict()

        return user_avg_returns, item_avg_returns

