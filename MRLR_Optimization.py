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
    # build Dc：item co-rated triples
    # accoding to R
    # return structure：user_id,item_id_one,item_id_two
    def init_Dc(self):
        R = self.R
        row = len(R)
        Dc = list()
        for i in range(row):
            non_zero = np.array(np.where(R[i] == 1))
            non_zero_size = non_zero.size
            if non_zero_size < 2:
                continue
            for j in range(non_zero_size - 1):
                u_vi = list([i, non_zero[0][j]])
                for k in range(j + 1, non_zero_size):
                    u_vi_copy = u_vi.copy()
                    u_vi_copy.append(non_zero[0][k])
                    u_vi_vk = np.array(u_vi_copy)
                    Dc.append(u_vi_vk)
        Dc = np.array(Dc)
        return Dc
    # build Dr：user-specific ranking triples
    # method：randomly sample
    # definition:for every Dc, there are 'Count' time Dr
    def init_Dr(self):
        Count = self.Count
        R = self.R
        row = len(R)
        Dr = list()
        for i in range(row):
            non_zero = np.array(np.where(R[i] == 1))
            is_zero = np.array(np.where(R[i] == 0))
            non_zero_size = non_zero.size
            is_zero_size = is_zero.size
            if non_zero_size < 2:
                continue
            for j in range(non_zero_size):
                u_vi = list([i, non_zero[0][j]])
                # 随机采样出count个树
                # sample_result = np.random.sample(range(is_zero_size),Count)
                sample_result = random.sample(range(is_zero_size), Count)
                for k in range(Count):
                    u_vi_copy = u_vi.copy()
                    sample_index = sample_result[k]
                    u_vi_copy.append(is_zero[0][sample_index])
                    u_vi_vj = np.array(u_vi_copy)
                    Dr.append(u_vi_vj)
        Dr = np.array(Dr)
        return Dr

    # negtive sampling
    # update parameter set
    # evalution

    # Negative Sampling Procedure
    # Draw N negative instances from distribution ~ Dr、Dc
    def init_NegativeSample(self):

        return None



