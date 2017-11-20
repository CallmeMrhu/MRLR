# author:hucheng

import MRLR_LoadData
import MRLR_Optimization
import MRLR_Optimization_run
import MRLR_Optimization_backup
import numpy as np

if __name__ == '__main__':
    # randomly init user_item
    # there are ten users and ten items
    # R means user_item
    # R = np.random.randint(0, 2, size=[10, 10])


    # randomly init item_category
    # there are ten categories for every item
    # C means item_category

    d = 10
    gamma_grid = {0.001,0.01,0.1,1.0}

    # R,C,lamda,alpha1,alpha3,gamma,d,iter,N,Count):

    # 测试
    loaddata = MRLR_LoadData.loadData()
    R, C, user_dict, asin_id, category_dict = loaddata.creat_R_C()
    # R, C, lamda, alpha1, alpha3, gamma, d, iter, N
    opt_MRLR = MRLR_Optimization_run.MRLR(R, C, 0.001, 0.8, 0.3, 0.01, 10, 50, 5)
    u,v,v_ctx,c = opt_MRLR.updata_parameter()

    # print(u)
    # print(v)

    # print(Dr)
    # print(Dr.shape)
