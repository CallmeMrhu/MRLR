# author : hucheng
# 2017.11.9

import numpy as np
import random
from scipy.stats import logistic

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
        self.alpha2 = 1-alpha1
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
                for id_one in range(non_zero_size - 1):
                    i = non_zero[0][id_one]
                    for id_two in range(i + 1, non_zero_size):
                        k = non_zero[0][id_two]
                        # calculate gradient
                        # update up
                        # P(vk|up,vi,sita)
                        kpi_up = v[k] * (1-logistic.cdf(self.alpha1 * np.dot(u[p],v[k])+self.alpha2 * np.dot(v[i],v[k])))
                        # P(vi|up,vk,sita)
                        ipk_up = v[i] * (1-logistic.cdf(self.alpha1 * np.dot(u[p],v[i])+self.alpha2 * np.dot(v[k],v[i])))
                        # negtive sampling:N
                        gpi_up = 0
                        gpk_up = 0
                        negtive_sample = random.sample(range(is_zero_size), N)
                        for index_three in range(N):
                            id_three = negtive_sample[index_three]
                            g = is_zero[0][id_three]
                            gpi_up += v[g] * (1 - logistic.cdf(-1 * (self.alpha1 * np.dot(u[p], v[g]) + self.alpha2 * np.dot(v[i], v[g]))))
                            gpk_up += v[g] * (1 - logistic.cdf(-1 * self.alpha1 * np.dot(u[p], v[k]) + self.alpha2 * np.dot(v[k], v[g])))
                        # calculate negtive sampling
                        u[p] = u[p] + self.gamma * (kpi_up + ipk_up - gpi_up - gpk_up)
                        # update vi
                        kpi_vi = kpi_up
                        ipk_vi = u[p] * (1-logistic.cdf(self.alpha1 * np.dot(u[p],v[k])+self.alpha2 * np.dot(v[i],v[k])))
                        gpi_vi = gpi_up
                        gpk_vi = 0
                        v[i] = v[i] + self.gamma * (kpi_vi + ipk_vi + gpi_vi + gpk_vi)
                        # update cl : category
                        # 待写
                    # number of Dr
                    number_Dr = non_zero_size
                    sample_result = random.sample(range(is_zero_size), number_Dr)
                    for sample_index in range(number_Dr):
                        id_four = sample_result[sample_index]
                        j = is_zero[0][id_four]
                        jpi_up = (v[i]-v[j]) * (1 - logistic.cdf(np.dot(u[p],v[i])-np.dot(u[p],v[j])))
                        gpj_up = 0
                        negtive_sample = random.sample(range(is_zero_size), N)
                        for index_three in range(N):
                            id_five = negtive_sample[index_three]
                            g = is_zero[0][id_five]
                            gpj_up += (v[g]-v[j])*(1 - logistic.cdf(-1 * (np.dot(u[p],v[j])-np.dot(u[p],v[g]))))
                        u[p] = u[p] + self.gamma * (jpi_up + gpj_up)
                        jpi_vi = u[p] * (1 - logistic.cdf(np.dot(u[p],v[i])-np.dot(u[p],v[j])))
                        v[i] = v[i] + self.gamma * (jpi_vi)

            # J has converged, and then break;
            # 待写




