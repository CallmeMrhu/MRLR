# author : hucheng

# target of this method: build
# matrix R：user * item
# matrix C：item * categories

import pandas as pd
import numpy as np
import gzip


class loadData():

    def parse(self, path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield eval(l)

    def getDF(self, path):
        i = 0
        df = {}
        for d in self.parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    def creat_R_C(self):
        # df_csv = pd.read_csv('ratings_Electronics.csv')

        # print(df[['asin','categories']].head(1))
        # print(df_csv.head(5))

        # category
        df_json = self.getDF('meta_Video_Games.json.gz')
        # df_json = getDF('meta_Electronics.json.gz')
        asin_category = df_json[['asin', 'categories']]
        # user - item
        df_csv = pd.read_csv('ratings_Video_Games.csv', header=None)
        # df_csv = pd.read_csv('ratings_Electronics.csv',header=None)
        df_csv.columns = ['user', 'asin', 'rating', 'timestamp']
        user_asin = df_csv.iloc[:10000]
        user_asin = user_asin[['user', 'asin']]

        # 提取数据集前N条数据 user-asin
        # 在这N条数据中，对每个asin，提取其category
        # 生成R、C
        user_dict = dict()
        user_num = 0
        asin_dict = dict()
        asin_num = 0

        user_list = list(user_asin['user'].drop_duplicates())
        item_list = list(user_asin['asin'].drop_duplicates())

        print('user_list:%d' % len(user_list))
        print('item_list:%d' % len(item_list))

        R = np.zeros((len(user_list), len(item_list)))

        for indexs in user_asin.index:
            user_id = 0
            asin_id = 0
            user = user_asin.loc[indexs].values[0]
            # print(user_asin.loc[indexs].values[0])
            asin = user_asin.loc[indexs].values[1]
            # print(user_asin.loc[indexs].values[1])
            if user not in user_dict.keys():
                user_dict[user] = user_num
                user_id = user_num
                user_num += 1

            else:
                user_id = user_dict[user]

            if asin not in asin_dict.keys():
                asin_dict[asin] = asin_num
                asin_id = asin_num
                asin_num += 1
            else:
                asin_id = asin_dict[asin]

            R[user_id][asin_id] = 1

        category_dict = dict()
        category_num = 0
        # 最后需要根据类别数目对矩阵提取切片
        C = np.zeros((len(item_list), 10000))

        for asin in item_list:
            asin_id = asin_dict[asin]
            category_id = 0
            result = asin_category[asin_category['asin'] == asin]
            index = result.index.tolist()[0]
            categories = asin_category.ix[index, 'categories']
            category_len = len(categories)
            for i in range(category_len):
                category_length = len(categories[i])
                for j in range(category_length):
                    category = (categories[i])[j]
                    if category not in category_dict.keys():
                        category_dict[category] = category_num
                        category_id = category_num
                        category_num += 1
                    else:
                        category_id = category_dict[category]
                    C[asin_id][category_id] = 1

        C = C[0:asin_num, 0:category_num]

        # print(user_num,asin_num,category_num)
        # print(C.shape)
        # print(R.shape)
        return R, C, user_dict, asin_id, category_dict
