import numpy as np
import torch
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class GMM:
    def __init__(self, dataset, num_clusters, converge=False, num_iterations=100):
        '''
        :param dataset: type: torch.Tensor
        :param num_clusters: type: int
        :param converge: if false check for convergence,
            otherwise use num_iterations, type: bool
        :param num_iterations: number of times to iterate EM, type: int
        mu: mean vector for each cluster. Select k points from the dataset
            to initialize mean vectors,
            type: list of tensors k*torch.Tensor(features)
        cov: list of covariance matrices for each cluster,
            initialized by setting each one of K covariance matrix equal to the
            covariance of the entire dataset
            type: list of tensors k*(torch.Tensor(features, features))
        priors: subjective priors assigned to clusters type: torch.Tensor(k)
        likelihoods: multivariate gaussian distributions
            type: torch.Tensor(num_samples, features)
        N: number of samples in the dataset, type: int
        '''
        self.X = dataset
        self.N = self.X.shape[0]
        self.d = self.X.shape[1]
        self.k = num_clusters
        self.converge = converge
        self.num_iterations = num_iterations
        # chooses k random indices out of N
        self.mu = self.X[torch.from_numpy(np.random.choice(self.N, self.k,
                                                           replace=False))].double()
        x_norm = self.X - torch.mean(self.X, 0).double()
        x_cov = ((x_norm.t() @ x_norm) / (self.N - 1)).double()
        self.cov = self.k * [x_cov]
        # self.cov = torch.ones((self.d, self.d))
        self.priors = torch.empty(self.k).fill_(1. / self.k).double()  # uniform priors
        self.likelihoods = torch.zeros(self.N * self.k).view(self.N, self.k).double()

    def e_step(self):
        '''
        Def:
        Computes the likelihoods and posteriors
        :return posteriors: type: torch.Tensor(num_samples, clusters)
        '''
        for i, data in enumerate(self.X):
            for j in range(self.k):
                self.likelihoods[i, j] = MultivariateNormal(self.mu[j],
                                                            self.cov[j]).log_prob(data).exp_().double()
        posteriors = torch.zeros(self.N * self.k).view(self.N, self.k).double()
        for i, data in enumerate(self.X):
            p_data = 0
            for j in range(self.k):
                p_data += (self.likelihoods[i, j] * self.priors[j])
            for j in range(self.k):
                posteriors[i, j] = (self.likelihoods[i, j] * self.priors[j]) / p_data
        return posteriors

    def m_step(self, posteriors, eps=1e-6, min_var=1e-3):
        '''
        Def:
        Sets new mean, covariance and prior to the Gaussian clusters
        :param posteriors: type: torch.Tensor(num_samples, clusters)
        :param eps: to avoid getting NaN, type: double
        :param min_var: type: double
        :return mu_updated: updated means,
            type: list of tensors k*torch.Tensor(features)
        :return cov_updated: updated covariance matrices,
            list of tensors k*(torch.Tensor(features, features))
        '''
        mu_updated = self.k * [torch.zeros(self.d).double()]
        mu_updated = torch.stack(mu_updated, dim=0)
        cov_updated = self.k * [torch.zeros((self.d, self.d)).double()]
        cov_updated = torch.stack(cov_updated, dim=0)
        norm = torch.sum(posteriors, dim=0) + eps  # normalizer
        for i in range(self.k):
            for j, data in enumerate(self.X):
                mu_updated[i] += data.double() * posteriors[j, i]
            mu_updated[i] = mu_updated[i] / norm[i]

        for i in range(self.k):
            for j, data in enumerate(self.X):
                cov_updated[i] += ((data.double() - mu_updated[i]).view(self.d, 1) @
                                   (data.double() - mu_updated[i]).view(1, self.d)) \
                                  * posteriors[j, i]
            cov_updated[i] = torch.clamp((cov_updated[i] / (norm[i])), min=min_var)
            self.priors[i] = norm[i] / self.N
        print("priors:\n", self.priors)
        for m in mu_updated:
            print("mean:\n", m)
        for c in cov_updated:
            print("covariance:\n", c)
        return mu_updated, cov_updated

    def convergence(self):
        '''
        tests for convergence
        :return mu: type: torch.Tensor (features)
        :return var: type: torch.Tensor (features)
        :return posteriors: type: torch.Tensor(num_samples, clusters)
        :return likelihoods: type: torch.Tensor(num_samples, features)
        '''
        if self.converge:
            # TODO a convergence method other than num_iterations
            pass
        else:
            for _ in tqdm(range(self.num_iterations)):
                posteriors = self.e_step()
                mu_updated, cov_updated = self.m_step(posteriors)
                self.mu, self.cov = mu_updated, cov_updated
        return self.mu, self.cov, posteriors, self.likelihoods


def density_plot(data, mu, cov, k, n=100):
    '''
    :param data: type: numpy.array(num_samples x features)
    :param mu: type: torch.Tensor (features)
    :param var: type: torch.Tensor (features)
    :param n: number of samples to generate, type: int
    :return likelihoods: type: torch.Tensor(num_samples, features)
    '''
    def vis(xx, yy, z):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, z, rstride=3, cstride=3, linewidth=1,
                        antialiased=True,
                        cmap=cm.inferno)
        cset = ax.contourf(xx, yy, z, zdir='z', offset=-0.15, cmap=cm.inferno)
        ax.set_zlim(-0.15, 0.2)
        ax.view_init(27, -21)
        plt.show()

    # Extract x and y
    x, y = data[:, 0], data[:, 1]
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    xx = np.linspace(xmin, xmax, n)
    yy = np.linspace(ymin, ymax, n)
    # Create meshgrid
    xx, yy = np.meshgrid(xx, yy)
    # get the design matrix
    samples = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
    samples = torch.from_numpy(samples).double()
    # compute the densities under each mixture
    likelihoods = torch.zeros((n**2, k)).double()
    for i, data in enumerate(samples):
        for j in range(k):
            likelihoods[i, j] = MultivariateNormal(mu[j], cov[j]).log_prob(data).exp_().double()
    # sum the densities to get mixture density
    likelihoods = torch.sum(likelihoods, dim=1).data.numpy().reshape([n, n])
    vis(xx, yy, likelihoods)

def main():
    '''
    Unit test on a synthetic dataset generated from 3 2D Gaussians
    '''
    def synthetic_data(mu, var, num_samples=500):
        """
        :param mu: type: torch.Tensor (features)
        :param var: type: torch.Tensor (features)
        :param num_samples: number of samples to be drawn, type: int
        :return: type: torch.Tensor (num_samples, features)
        """
        data = []
        for i in range(num_samples):
            data += [torch.normal(mu, var.sqrt())]
        return torch.stack(data, dim=0)

    # generate some clusters (uneven number of samples to test robustness)
    n1, n2, n3 = 300, 500, 1000
    cluster1 = synthetic_data(torch.Tensor([2.5, 2.5]), torch.Tensor([1.2, .8]),
                              num_samples=n1).double()
    cluster2 = synthetic_data(torch.Tensor([7.5, 7.5]), torch.Tensor([.75, .5]),
                              num_samples=n2).double()
    cluster3 = synthetic_data(torch.Tensor([8, 15]), torch.Tensor([.6, .8]),
                              num_samples=n3).double()
    x = torch.cat([cluster1, cluster2, cluster3]).double()
    plt.scatter(x.numpy()[:n1, 0], x.numpy()[:n1, 1], color='red')
    plt.scatter(x.numpy()[n1:n1+n2, 0], x.numpy()[n1:n1+n2, 1], color='blue')
    plt.scatter(x.numpy()[n1+n2:, 0], x.numpy()[n1+n2:, 1], color='green')
    plt.title('Generated Data from 3 2D Gaussians')
    plt.show()
    gmm = GMM(x, num_clusters=3, num_iterations=50)
    mu, cov, posteriors, likelihoods = gmm.convergence()
    density_plot(x.numpy(), mu, cov, gmm.k)


if __name__ == "__main__":
    main()
