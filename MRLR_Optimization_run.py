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
    def __init__(self, R, C, lamda, alpha1, alpha3, gamma, d, iter, N):
        self.R = R
        self.C = C
        self.lamda = lamda
        self.alpha1 = alpha1
        self.alpha2 = 1 - alpha1
        self.alpha3 = alpha3
        self.gamma = gamma
        self.d = d
        self.iter = iter
        self.N = N

    # init user embedding
    def init_User_Embedding(self):
        user_number = len(self.R)
        u = np.random.random((user_number, self.d))
        # print(u)
        return u

    # init item embedding
    def init_Item_Embedding(self):
        item_number = len(self.R[0])
        v = np.random.random((item_number, self.d))
        # print(v)
        return v

    # init category embedding
    def init_Categories_Embedding(self):
        category_number = len(self.C[0])
        c = np.random.random((category_number, self.d))
        return c

    # add category embedding to item embedding
    def add_Categories_to_Item(self, v, c):
        for p in range(len(self.R)):
            # every user has many items
            non_zero_item_up = np.array(np.where(self.R[p] == 1))
            non_zero_size_item_up = non_zero_item_up.size
            if non_zero_size_item_up == 0:
                continue
            for id_one in range(non_zero_size_item_up):
                i = non_zero_item_up[0][id_one]
                # every item belongs to many categories
                non_zero_category_vi = np.array(np.where(self.C[i] == 1))
                non_zero_size_category_vi = non_zero_category_vi.size
                parameter = self.alpha3 / float(non_zero_size_category_vi)
                for id_two in range(non_zero_size_category_vi):
                    cl = non_zero_category_vi[0][id_two]
                    # at the same time ,uodate v[i]
                    v[i] = v[i] + parameter * c[cl]
        return v

    def updata_parameter(self):
        u = self.init_User_Embedding()
        v = self.init_Item_Embedding()
        c = self.init_Categories_Embedding()
        v = self.add_Categories_to_Item(v, c)
        R = self.R
        C = self.C
        N = self.N

        # s = self.calcResult(u, v)
        # print(s)
        for t in range(self.iter):
            # every User p
            for p in range(len(R)):
                # every user has many items
                non_zero_item_up = np.array(np.where(R[p] == 1))
                # test = np.where(R[p] == 0)
                # is_zero_item_up = test[0].tolist()
                is_zero_item_up = np.array(np.where(R[p] == 0))
                non_zero_size_item_up = non_zero_item_up.size
                is_zero_size_item_up = is_zero_item_up.size
                if non_zero_size_item_up < 2:
                    continue
                for id_one in range(non_zero_size_item_up - 1):
                    i = non_zero_item_up[0][id_one]
                    # every item belongs to many categories
                    non_zero_category_vi = np.array(np.where(C[i] == 1))
                    non_zero_size_category_vi = non_zero_category_vi.size
                    # mistake : i should be id_one
                    for id_two in range(id_one + 1, non_zero_size_item_up):
                        k = non_zero_item_up[0][id_two]
                        non_zero_category_vk = np.array(np.where(C[k] == 1))
                        non_zero_size_category_vk = non_zero_category_vk.size
                        # calculate gradient

                        # P(vk|up,vi,sita)
                        kpi_up = v[k] * (
                            1 - logistic.cdf(self.alpha1 * np.dot(u[p], v[k]) + self.alpha2 * np.dot(v[i], v[k])))
                        # P(vi|up,vk,sita)
                        ipk_up = v[i] * (
                            1 - logistic.cdf(self.alpha1 * np.dot(u[p], v[i]) + self.alpha2 * np.dot(v[k], v[i])))
                        # negtive sampling:N
                        # calculate negtive sampling
                        gpi_up = 0
                        gpk_up = 0
                        negtive_sample = random.sample(range(is_zero_size_item_up), N)
                        for index_three in range(N):
                            id_three = negtive_sample[index_three]
                            g = is_zero_item_up[0][id_three]
                            gpi_up = gpi_up + v[g] * (1 - logistic.cdf(
                                (-1) * (self.alpha1 * np.dot(u[p], v[g]) + self.alpha2 * np.dot(v[k], v[g]))))

                            gpk_up = gpk_up + v[g] * (1 - logistic.cdf(
                                (-1) * (self.alpha1 * np.dot(u[p], v[g]) + self.alpha2 * np.dot(v[i], v[g]))))

                            # 2017.11.12 16:00
                            # update negtive item vg, this is neccessary
                            v[g] = v[g] - self.gamma * (u[p] + v[i]) * (1 - logistic.cdf(
                                (-1) * (self.alpha1 * np.dot(u[p], v[g]) + self.alpha2 * np.dot(v[i], v[
                                    g])))) - 2 * self.lamda * np.sqrt(np.dot(v[g], v[g]))
                        # update up
                        gradient_Dc_up = kpi_up + ipk_up - gpi_up - gpk_up
                        u[p] = u[p] + self.gamma * gradient_Dc_up - 2 * self.lamda * np.sqrt(
                            np.dot(u[p], u[p]))
                        # update vi
                        kpi_vi = kpi_up
                        ipk_vi = (u[p] + v[k]) * (
                            1 - logistic.cdf(self.alpha1 * np.dot(u[p], v[k]) + self.alpha2 * np.dot(v[k], v[i])))
                        gpi_vi = 0
                        gpk_vi = gpk_up
                        gradient_Dc_vi = kpi_vi + ipk_vi - gpi_vi - gpk_vi
                        v[i] = v[i] + self.gamma * gradient_Dc_vi - 2 * self.lamda * np.sqrt(
                            np.dot(v[i], v[i]))
                        # update vk
                        ipk_vk = ipk_up
                        kpi_vk = (u[p] + v[i]) * (
                            1 - logistic.cdf(self.alpha1 * np.dot(u[p], v[k]) + self.alpha2 * np.dot(v[i], v[k])))
                        gpk_vk = 0
                        gpi_vk = gpi_up
                        gradient_Dc_vk = ipk_vk + kpi_vk - gpk_vk - gpi_vk
                        v[k] = v[k] + self.gamma * gradient_Dc_vk - 2 * self.lamda * np.sqrt(
                            np.dot(v[k], v[k]))

                        # update cl : category,including Cvi & Cvk
                        # as to Cvi

                        # parameter = self.alpha3 / float(non_zero_size_category_vi)
                        # cvil = np.zeros(self.d)
                        # for index_four in range(non_zero_size_category_vi):
                        #     cl = non_zero_category_vi[0][index_four]
                        #     c[cl] = c[cl] + self.gamma * parameter * (u[p] + v[k]) * (
                        #         1 - logistic.cdf(self.alpha1 * np.dot(u[p], v[i]) + self.alpha2 * np.dot(v[k], v[i])))
                        #     cvil += c[cl]
                        # v[i] = v[i] + parameter * cvil

                        # as to Cvk

                        # parameter = self.alpha3 / float(non_zero_size_category_vk)
                        # cvkl = np.zeros(self.d)
                        # for index_five in range(non_zero_size_category_vk):
                        #     cl = non_zero_category_vk[0][index_five]
                        #     c[cl] = c[cl] + self.gamma * parameter * (u[p] + v[i]) * (
                        #         1 - logistic.cdf(self.alpha1 * np.dot(u[p], v[k]) + self.alpha2 * np.dot(v[i], v[k])))
                        #     cvkl = cvkl + c[cl]
                        # v[k] = v[k] + parameter * cvkl

                # update Dr
                for id_one in range(non_zero_size_item_up):
                    i = non_zero_item_up[0][id_one]
                    # number of Dr
                    # identify number of vi == number of vj
                    number_Dr = non_zero_size_item_up
                    sample_result = random.sample(range(is_zero_size_item_up), number_Dr)
                    for sample_index in range(number_Dr):
                        id_four = sample_result[sample_index]
                        j = is_zero_item_up[0][id_four]
                        jpi_up = (v[i] - v[j]) * (1 - logistic.cdf(np.dot(u[p], v[i]) - np.dot(u[p], v[j])))
                        jpi_vj = (-1) * u[p] * (1 - logistic.cdf(np.dot(u[p], v[i]) - np.dot(u[p], v[j])))
                        gpj_up = 0
                        gpj_vj = 0
                        negtive_sample = random.sample(range(is_zero_size_item_up), N)
                        for index_three in range(N):
                            id_five = negtive_sample[index_three]
                            g = is_zero_item_up[0][id_five]
                            # if j == g:
                            #     # print('randomly sampling at the same position')
                            #     continue
                            gpj_up = gpj_up + (v[g] + v[j]) * (
                                1 - logistic.cdf((-1) * (np.dot(u[p], v[j]) + np.dot(u[p], v[g]))))
                            # 2017.11.12 16:00
                            # update negtive item vg and vj, this is neccessary
                            gpj_vj = gpj_vj + u[p] * (1 - logistic.cdf((-1) * (np.dot(u[p], v[j]) + np.dot(u[p], v[g]))))
                            v[g] = v[g] - self.gamma * u[p] * (
                                1 - logistic.cdf(
                                    (-1) * (np.dot(u[p], v[j]) + np.dot(u[p], v[g])))) - 2 * self.lamda * np.sqrt(
                                np.dot(v[g], v[g]))

                        # update up
                        gradient_Dr_up = jpi_up - gpj_up
                        u[p] = u[p] + self.gamma * gradient_Dr_up - 2 * self.lamda * np.sqrt(
                            np.dot(u[p], u[p]))
                        # update vi
                        gradient_Dr_vi = u[p] * (1 - logistic.cdf(np.dot(u[p], v[i]) - np.dot(u[p], v[j])))
                        v[i] = v[i] + self.gamma * gradient_Dr_vi - 2 * self.lamda * np.sqrt(np.dot(v[i], v[i]))
                        # 2017.11.12 16:00
                        # update negtive item vg and vj, this is neccessary
                        # update vj
                        gradient_Dr_vj = jpi_vj - gpj_vj
                        v[j] = v[j] + self.gamma * gradient_Dr_vj - 2 * self.lamda * np.sqrt(np.dot(v[j], v[j]))

                        # as to Cvi
                        # parameter = self.alpha3 / float(non_zero_size_category_vi)
                        # cvil = np.zeros(self.d)
                        # for index_six in range(non_zero_size_category_vi):
                        #     cl = non_zero_category_vi[0][index_six]
                        #     c[cl] = c[cl] + self.gamma * parameter * u[p] * (
                        #         1 - logistic.cdf(np.dot(u[p], v[i]) - np.dot(u[p], v[j])))
                        #     cvil += c[cl]
                        # v[i] = v[i] + parameter * cvil

            J = self.calcResult(u, v)
            print('J:%f' % J)
            print('iter:%d' % t)
            if J <= 0.05:
                break

        return u, v, c

    def calcResult(self, u, v):
        # J has converged, and then break;
        pDc = 0
        pDr = 0
        for p in range(len(self.R)):
            # every user has many items
            non_zero_item_up = np.array(np.where(self.R[p] == 1))
            is_zero_item_up = np.array(np.where(self.R[p] == 0))
            non_zero_size_item_up = non_zero_item_up.size
            is_zero_size_item_up = is_zero_item_up.size
            if non_zero_size_item_up < 2:
                continue
            for id_one in range(non_zero_size_item_up - 1):
                i = non_zero_item_up[0][id_one]
                for id_two in range(id_one + 1, non_zero_size_item_up):
                    k = non_zero_item_up[0][id_two]
                    # m = self.alpha1 * np.dot(u[p], v[k]) + self.alpha2 * np.dot(v[i], v[k])
                    pvk_positive = logistic.cdf(self.alpha1 * np.dot(u[p], v[k]) + self.alpha2 * np.dot(v[i], v[k]))
                    pvi_positive = logistic.cdf(self.alpha1 * np.dot(u[p], v[i]) + self.alpha2 * np.dot(v[k], v[i]))
                    negtive_sample = random.sample(range(is_zero_size_item_up), self.N)
                    pvk_negtive = 1
                    pvi_negtive = 1
                    for index_three in range(self.N):
                        # for index_three in range(4):
                        id_three = negtive_sample[index_three]
                        g = is_zero_item_up[0][id_three]
                        pvk_negtive = pvk_negtive * logistic.cdf(
                            (-1) * (self.alpha1 * np.dot(u[p], v[g]) + self.alpha2 * np.dot(v[i], v[g])))

                        pvi_negtive = pvi_negtive * logistic.cdf(
                            (-1) * (self.alpha1 * np.dot(u[p], v[g]) + self.alpha2 * np.dot(v[k], v[g])))

                    pvk = pvk_positive * pvk_negtive
                    pvi = pvi_positive * pvi_negtive
                    # print(pvk)
                    # print(pvi)
                    result = pvk * pvi
                    # very important!!!
                    if result != 0:
                        pDc += np.log(result)
                    else:
                        print("there is an error_one,pvk:%f , pvi:%f " % (pvk,pvi))
                        # print(pDc)

            for id_one in range(non_zero_size_item_up):
                i = non_zero_item_up[0][id_one]
                number_Dr = non_zero_size_item_up
                sample_result = random.sample(range(is_zero_size_item_up), number_Dr)
                for sample_index in range(number_Dr):
                    id_four = sample_result[sample_index]
                    j = is_zero_item_up[0][id_four]

                    pij_postive = logistic.cdf(np.dot(u[p], v[i]) - np.dot(u[p], v[j]))
                    # print('pij_postive:%f' % pij_postive)
                    pij_negtive = 1
                    negtive_sample = random.sample(range(is_zero_size_item_up), self.N)
                    for index_three in range(self.N):
                        # for index_three in range(4):
                        id_five = negtive_sample[index_three]
                        g = is_zero_item_up[0][id_five]
                        pij_negtive = pij_negtive * logistic.cdf(-1 * (np.dot(u[p], v[j]) + np.dot(u[p], v[g])))

                    result = pij_postive * pij_negtive
                    if result != 0:
                        pDr += np.log(result)
                    else:
                        print("there is an error_two,pij_postive:%f , pij_negtive:%f " % (pij_postive,pij_negtive))

        J = (pDc + pDr) * (-1)
        print('--------------------------------pDc+pDr = %f' % J)
        reg = 0
        for user_id in range(len(u)):
            reg += np.dot(u[user_id], u[user_id])
        for item_id in range(len(v)):
            reg += np.dot(v[item_id], v[item_id])

        J += self.lamda * np.sqrt(reg)
        return J
