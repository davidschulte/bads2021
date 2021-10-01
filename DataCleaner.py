import pandas as pd
import numpy as np

class DataCleaner:

    def __init__(self):
        self.cat_features = ['item_id', 'item_size', 'item_color', 'brand_id', 'user_id', 'user_title', 'user_state']
        self.int_features = []
        self.float_features = ['item_price']
        self.date_features = ['order_date', 'delivery_date', 'user_dob', 'user_reg_date']
        self.target = 'return'


    def convert_data_types(self, data):
        for feature in self.cat_features:
            data[feature] = data[feature].astype('category')
        for feature in self.date_features:
            data[feature] = pd.to_datetime(data[feature])

        #data[self.target] = data[self.target].astype('bool')

        return data

    def create_features_oneline(self, data):
        data['delivery_span'] = (data['delivery_date'] - data['order_date']).dt.days
        data['user_age'] = (data['order_date'] - data['user_dob']).dt.days / 365

        month_days = []
        for day in data['order_date']:
            month_days.append(day.day)

        data['order_month_days'] = month_days

        #weekday!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        return data

    def create_features_multiline(self, data):
        data['order_num_items'] = data.groupby(['user_id', 'order_date'])['user_id'].transform('size').astype(
            np.int32)

        data['order_num_spec_item'] = data.groupby(['user_id', 'order_date', 'item_id'])['user_id'].transform(
            'size').astype(np.int32)

        data['order_sum'] = data.groupby(['user_id', 'order_date'])['item_price'].transform('sum').astype(np.float32)

        data['user_orders'] = data.groupby(['user_id'])['user_id'].transform('size').astype(np.float32)

        data['user_return_rate'] = data.groupby(['user_id'])['return'].transform('mean').astype(np.float32)

        data['item_popularity'] = data.groupby(['item_id'])['item_id'].transform('size').astype(np.int32)

        data['brand_popularity'] = data.groupby(['brand_id'])['brand_id'].transform('size').astype(np.int32)

        data['brand_discount'] = data.groupby(['brand_id'])['item_price'].transform('mean').astype(np.float32) - data[
            'item_price']

        data['item_color_popularity'] = data.groupby(['item_color'])['item_color'].transform('size').astype(np.int32)

        data['item_return_rate'] = data.groupby(['item_id'])['return'].transform('mean').astype(np.float32)

        data['color_return_rate'] = data.groupby(['item_color'])['return'].transform('mean').astype(np.float32)

        return data
    
    def clean(self, data):
        data.loc[data['user_dob'] < "1920-01-01", 'user_dob'] = np.nan
        data.loc[data['user_dob'] > "2010-01-01", 'user_dob'] = np.nan
        data.loc[data['user_dob'].isnull(), 'user_dob'] = data['user_dob'].quantile(0.5, interpolation="midpoint")

        data.loc[data['delivery_span'].isnull(), 'delivery_span'] = data['delivery_span'].quantile(0.5, interpolation="midpoint")

        return data

    def change_after_multiline(self, data):
        data.drop(self.cat_features+self.date_features, axis=1, inplace=True)

        return data

    from scipy.stats import chi2_contingency

    def optimize_grouping(self, cat_feature, target_feature):
        '''
        Compares differenct encodings of a categorical variable using Chi^2 test.

        Input:
        - cat_feature: categorical feature to be encoded
        - target_feature: target feature

        Output:
        - vector of Chi^2 statistic values
        '''

        # Copying features to avoid editing the original DataFrame
        cat_feature = cat_feature.copy()
        target_feature = target_feature.copy()

        # Checking if feature is categorical
        if cat_feature.dtype != 'category':
            print('Input feature is not categorical. Received feature of type:', cat_feature.dtype)
            return

        # Placeholders for Chi^2 values and categories
        stats = []
        cats = []
        cats_num = []

        # Storing number and values of categories
        n_unique = cat_feature.nunique()
        cats_num.append(n_unique)
        cats.append(cat_feature.cat.categories)

        # Performing chi2 test
        ct = pd.crosstab(cat_feature, target_feature)
        stat, _, _, _ = chi2_contingency(ct)
        stats.append(stat)

        # Iteratively trying different groupings
        for i in range(n_unique - 1):

            # Computing odds ratio
            ct = pd.crosstab(cat_feature, target_feature)
            ct['odds_ratio'] = ct[0] / ct[1]

            # Finding min odds ratio difference
            ct = ct.sort_values('odds_ratio')
            ct['odds_ratio_diff'] = ct['odds_ratio'].diff()
            min_idx = np.where(ct['odds_ratio_diff'] == ct['odds_ratio_diff'].min())[0][0]

            # Storing levels to merge
            levels_to_merge = ct.iloc[(min_idx - 1):(min_idx + 1)].index.values

            # Merging two categories with add_categories()
            cat_feature.cat.add_categories(['+'.join(str(levels_to_merge))], inplace=True)
            for level in levels_to_merge:
                cat_feature.loc[cat_feature == level] = '+'.join(str(levels_to_merge))
                cat_feature.cat.remove_categories([level], inplace=True)

            # Storing number and values of categories after encoding
            cats_num.append(cat_feature.nunique())
            cats.append(cat_feature.cat.categories)

            # Performing chi2 test
            ct = pd.crosstab(cat_feature, target_feature)
            stat, _, _, _ = chi2_contingency(ct)
            stats.append(stat)

        # Plotting results
        import matplotlib.pyplot as plt
        plt.plot(cats_num, stats)
        plt.title('Chi^2 Elbow Curve')
        plt.ylabel('Chi^2 statistic')
        plt.xlabel('Number of categories')
        plt.show()

        # Printing encodings
        for i in range(len(cats)):
            print('- {} categories: {}'.format(cats_num[i], cats[i].values))

        # Returning Chi^2 values and encodings
        return stats, cats