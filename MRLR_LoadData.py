# author : hucheng

# target of this method: build
# matrix R：user * item
# matrix C：item * categories

import pandas as pd
import numpy as np
import gzip


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


if __name__ == '__main__':
    # df_csv = pd.read_csv('ratings_Electronics.csv')

    # print(df[['asin','categories']].head(1))
    # print(df_csv.head(5))


    df = getDF('meta_Electronics.json.gz')
    df_categories = df[['categories']]

    print(type(df_categories))

    nparray_catagories = np.array(df_categories)

    print(type(nparray_catagories))

    size = nparray_catagories.shape
    print(size)

    # i = 0
    # while df_categories.iloc(i):
    #
    #     print(df_categories.iloc(i))
    #     print(type(df_categories.iloc(i)))
    #     row = np.array(df_categories.iloc(i))
    #     row = row.tolist()
    #     print(row)
    #     break
    #     # i += 1


    # print(np.array(df[['categories']].head(5)))
