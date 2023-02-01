import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.linalg import eig
import math

GAMMA = 0.4
T2 = 0.6
N = 40
epsilon = 0.0001


def diffusion_mat(datas, gaussian_kfunc):
    """
    Parameters
    ----------
    datas : ndarray
        n samples with m features should have shape [n,m].
    gaussian_kfunc: callable function
        gaussian kernel function.
    Returns
    -------
    ndarray
        one-step difussion matrix.
    """

    df_mat = np.zeros((datas.shape[0] ,datas.shape[0]))
    for i in range(datas.shape[0]):
        for j in range( i +1 ,datas.shape[0]):
            df_mat[i ,j] = gaussian_kfunc(datas[i], datas[j])
    df_mat = df_mat + df_mat.T + np.eye(datas.shape[0], datas.shape[0])  # calculate the 'distance', same values result 1
    return df_mat


class GaussianKernel:
    def __init__(self, N, epsilon, norm="L1"):
        self.N = N
        self.epsilon = epsilon
        self.norm = norm

    def __call__(self, x1, x2):
        if self.norm == "L1":
            return np.exp(-np.sum(np.abs(x1 -x2) )** 2 /( 2 *self.epsilon *self.N**2))
        elif self.norm == "L2":
            return np.exp(-np.abs(x1 - x2)**2 / (2 * self.epsilon * self.N**2))
        else:
            raise Exception("undefine norm type")


class SSH_OB_data:
    def __init__(self, gamma, t1_list, t2, N, path="./"):
        self.path = path
        self.gamma = gamma
        self.t1_list = t1_list
        self.t2 = t2
        self.N = N

    def construct_p(self, t1):
        U = np.array([[0, t1+self.gamma], [t1-self.gamma, 0]])
        T = np.array([[0, 0], [self.t2, 0]])
        init = U
        for cell in range(self.N-1):
            init = la.block_diag(init, U)
        for row in range(1, self.N-1):
            init[2*row:2*row+2, 2*row-2:2*row] = (T.transpose().conjugate())
            init[2 * row:2 * row+2, 2 * row+2:2 * row + 4] = T
        init[0:2, 2:4] = T
        init[2*self.N-2:2*self.N, 2*self.N-4:2*self.N-2] = T.transpose().conjugate()

        w1, right_v = np.linalg.eig(init)  # w: eigenvalue; v: eigenstate. Complex Hermitian (conjugate symmetric) or a real symmetric matrix.
        w2, left_v = np.linalg.eig(
                 init.transpose().conjugate())  # w: eigenvalue; v: eigenstate. Complex Hermitian (conjugate symmetric) or a real symmetric matrix.
        w2 = w2.conjugate()
        inv_index = np.argsort(w2)
        # inv_index = inv_index[::-1]
        w2 = w2[inv_index]
        left_v = left_v[:, inv_index]
        inv_index = np.argsort(w1)
        w1 = w1[inv_index]
        right_v = right_v[:, inv_index]

        '''Energy'''
        num_list1 = []
        num_list2 = []
        for i in range(right_v.shape[1]):
            if w1[i].real < -0.001:
                num_list1.append(i)
        for i in range(left_v.shape[1]):
            if w2[i].real < -0.001:
                num_list2.append(i)
        right_v = right_v[:, num_list1]
        w1 = w1[num_list1]
        left_v = left_v[:, num_list2]
        w2 = w2[num_list2]

        # w1, left_v, right_v = eig(init, left=True)
        test = np.dot(left_v.transpose().conjugate(), right_v)
        aa = right_v.shape[0]
        for i in range(right_v.shape[1]):
            if np.sqrt(test[i, i])>10**(-12):
                right_v[:, i] = right_v[:, i] / np.sqrt(test[i, i])
                left_v[:, i] = left_v[:, i] / np.sqrt(test[i, i])
        test2 = np.dot(left_v.transpose().conjugate(), right_v)

        # left_v = left_v.transpose().conjugate()

        # test = np.dot(left_v.transpose().conjugate(), right_v)
        # remove_list = []
        # for i in range(len(w1)):
        #     if abs(w1[i]) <= 0.001:
        #         remove_list.append(i)
        # right_v = np.delete(right_v, remove_list, axis=1)
        # left_v = np.delete(left_v, remove_list, axis=1)
        # w1 = np.delete(w1, remove_list, axis=0)

        # num_list1 = []
        # num_list2 = []
        # for i in range(right_v.shape[1]):
        #     if w1[i].real < -0.001:
        #         num_list1.append(i)
        # for i in range(left_v.shape[1]):
        #     if w2[i].real < -0.001:
        #         num_list2.append(i)
        # right_v = right_v[:, num_list1]
        # left_v = left_v[:, num_list2]

        # test = list(range(1, self.N, 2))
        # test_0 = left_v.shape[1]

        # right_v = right_v[:, range(0, self.N-1, 2)]
        # left_v = left_v[:, range(1, self.N, 2)]

        p = np.dot(right_v, left_v.transpose().conjugate())
        # p = np.outer(right_v, left_v.transpose().conjugate())
        return p

    def modify_p(self, data):
        new_p = []
        for row in range(0, data.shape[0], 2):
            new_p.append(data[row, row+1])

        # num = 40
        # row_list_up = []
        # row_list_down = []
        # for row in range(0, num):
        #     row_list_up.append(row)
        #     row_list_down.append(2*self.N-1-row)
        # for row in row_list_up:
        #     for j in range(row, num):
        #         data[row, data.shape[0]-num+j] = 0
        # for row in row_list_down:
        #     for j in range(0, num-data.shape[0]+row+1):
        #         data[row, j] = 0

        # new = data
        new = np.array(new_p)
        return new


    def construct_p_list(self):
        """
        Returns
        -------
        ndarray
        [number of items * 2N * 2N]: the 3D structure for the projection matrix
        """
        p_list = []
        p_shape_list = []
        for t1_init in self.t1_list:
            initial_P = self.construct_p(t1_init)
            # p_shape_list.append(self.modify_p(initial_P).shape[0])
            # p_list.append(initial_P)
            a = self.modify_p(initial_P)
            p_list.append(self.modify_p(initial_P))
        p_list = np.array(p_list)
        # p_list = (p_list.transpose()/p_list.max(axis=1)).transpose()
        return p_list


if __name__ == '__main__':
    t1_list = []
    a = math.pi
    for i in range(500):
        t1_list.append(0.0 + i * 1 / math.pi ** 5)
        # t1_list.append(0.4902 + i * 1/math.pi**5)
    # t1_list = np.arange(0, 1.7, 0.001)
    test = SSH_OB_data(GAMMA, t1_list, T2, N)
    P_list = test.construct_p_list()
    kfunc = GaussianKernel(N, epsilon)
    df_mat = diffusion_mat(P_list, kfunc)  # first time diffusion

    df_prob = df_mat / np.sum(df_mat, axis=1)[:, np.newaxis]
    evals, evs = la.eig(df_prob)
    sorted_index = np.argsort(np.real(evals))[::-1]
    evs = evs[:, sorted_index]
    evals = evals[sorted_index]

    fig = plt.figure(figsize=(25, 4))
    ax1 = fig.add_subplot(151)
    ax1.set_title("Gaussian Kernel", fontsize=16)
    c = ax1.pcolormesh(df_mat)
    ax1.invert_yaxis()
    plt.colorbar(c)

    ax2 = fig.add_subplot(152)
    ax2.scatter(np.arange(20), evals[:20])
    ax2.set_title("the 10 largest eigen values", fontsize=16)

    ax3 = fig.add_subplot(153)
    ax3.scatter(np.arange(len(t1_list)), np.round(evs[:, 0], 3))
    ax3.set_title("$\Psi_0$", fontsize=16)

    ax4 = fig.add_subplot(154)
    ax4.scatter(np.arange(len(t1_list)), evs[:, 1])
    ax4.set_title("$\Psi_1$", fontsize=16)

    ax5 = fig.add_subplot(155)
    ax5.scatter(np.arange(len(t1_list)), evs[:, 2])
    ax5.set_title("$\Psi_2$", fontsize=16)
    plt.show()

    plt.scatter(evs[:, 0], evs[:, 1])
    # plt.title("$\psi_1 - \psi_4$")
    plt.xlabel("$\psi_1$")
    plt.ylabel("$\psi_1$")
    plt.show()
