# author:hucheng

import MRLR_LoadData
import MRLR_Optimization
import numpy as np
import random

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

    # 实验一下已经写好的代码，生成Dc,Dr...待做
    opt_MRLR = MRLR_Optimization.MRLR(R,C,0.001,0.001,0.001,0.001,8,100,4,2)
    Dc = opt_MRLR.init_Dc()
    print(Dc)
    Dr = opt_MRLR.init_Dr()
    print(Dr)
