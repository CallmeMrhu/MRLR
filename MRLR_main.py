# author:hucheng

import MRLR_LoadData
import MRLR_Optimization
import MRLR_Optimization_own
import numpy as np

if __name__ == '__main__':
    # randomly init user_item
    # there are ten users and ten items
    # R means user_item
    # R = np.random.randint(0, 2, size=[10, 10])

    R = [
        [0,1,1,0,0,0,0,1,0,0],
        [1,0,1,0,0,1,1,0,0,0],
        [1,0,1,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,0,1,0,0,0],
        [1,0,0,0,0,0,0,1,0,0],
        [1,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1],
        [0,0,1,0,0,0,0,1,0,1],
        [0,0,0,1,0,1,0,0,0,1]
    ]

    R = np.array(R)
    # randomly init item_category
    # there are ten categories for every item
    # C means item_category
    C = np.random.randint(0,2,size=[10,10])
    d = 10
    gamma_grid = {0.001,0.01,0.1,1.0}

    # R,C,lamda,alpha1,alpha3,gamma,d,iter,N,Count):

    # 测试
    opt_MRLR = MRLR_Optimization_own.MRLR(R,C,0.001,0.001,0.001,0.001,8,10,4,2)
    u,v,c = opt_MRLR.updata_parameter()
    # print(u)

    # print(Dr)
    # print(Dr.shape)
