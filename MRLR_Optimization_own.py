# author : hucheng
# 2017.11.9

import numpy as np
import random

# dataset structure：
# matrix R：user * item
# matrix C：item * categories

class MRLR():
    # init parameter set
    def __init__(self,R,C,lamda,alpha1,alpha3,gamma,d,iter,N,Count):
        self.R = R
        self.C = C
        self.lamda = lamda
        self.alpha1 = alpha1
        self.alpha3 = alpha3
        self.gamma = gamma
        self.d = d
        self.iter = iter
        self.N = N
        self.Count = Count
    # init user embedding
    def init_User_Embedding(self):
        user_number = len(self.R)
        u = np.random.random((user_number,self.d))
        return u
    # init item embedding
    def init_Item_Embedding(self):
        item_number = len(self.R[0])
        v = np.random.random((item_number,self.d))
        return v
    # init category embedding
    def init_Categories_Embedding(self):
        category_number = len(self.C[0])
        c = np.random.random((category_number,self.d))
        return c

    def updata_parameter(self):
        u = self.init_User_Embedding()
        v = self.init_Item_Embedding()
        c = self.init_Categories_Embedding()
        R = self.R
        N = self.N
        for t in range(self.iter):
            # every User p
            for p in range(len(R)):
                non_zero = np.array(np.where(R[p] == 1))
                is_zero = np.array(np.where(R[p] == 0))
                non_zero_size = non_zero.size
                is_zero_size = is_zero.size
                if non_zero_size < 2:
                    continue
                for i in range(non_zero_size - 1):
                    # u_vi = list([i, non_zero[0][j]])
                    for k in range(i + 1, non_zero_size):
                        # negtive sampling:N
                        negtive_sample = random.sample(range(is_zero_size), N)
                        for n in range(N):
                            g = negtive_sample[n]
                        # calculate negtive sampling
                        # calculate gradient
                        # P(vk|up,vi,sita)
                        # P(vi|up,vk,sita)
                        # update up
                        # update vi
                        # update cl



                    # number of Dr
                    number_Dr = non_zero_size
                    sample_result = random.sample(range(is_zero_size), number_Dr)
                    for sample_index in range(number_Dr):
                        j = sample_result[sample_index]

                        negtive_sample = random.sample(range(is_zero_size), N)
                        for n in range(N):
                            g = negtive_sample[n]

                            # calculate negtive sampling
                            # calculate gradient
                            # P(vj,vi|up,sita)
                            # update up
                            # update vi
                            # update cl





