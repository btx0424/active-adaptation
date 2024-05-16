import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNormalGamma(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.LazyLinear(dim * 4)
    
    def forward(self, input):
        mu, logv, logalpha, logbeta = self.linear(input).chunk(4, dim=-1)
        v = F.softplus(logv)
        alpha = F.softplus(logalpha) + 1
        beta = F.softplus(logbeta)
        return torch.cat([mu, v, alpha, beta], dim=-1)


def nig_nll(y, gamma, v, alpha, beta):
    twoBlambda = 2 * beta * (1+v)
    nll = 0.5 * torch.log(torch.pi/v)  \
        - alpha * torch.log(twoBlambda)  \
        + (alpha+0.5) * torch.log(v*(y-gamma)**2 + twoBlambda)  \
        + torch.lgamma(alpha)  \
        - torch.lgamma(alpha+0.5)
    return nll


def nig_reg(y, gamma, v, alpha, beta):
    error = torch.abs(y - gamma)
    evi = 2 * v + alpha
    reg = error * evi
    return reg


def evidential_regression(
    y_true: torch.Tensor, 
    evidential_output: torch.Tensor, 
    coeff: float=1.0
):
    gamma, v, alpha, beta = evidential_output.chunk(4, -1)
    loss_nll = nig_nll(y_true, gamma, v, alpha, beta)
    loss_reg = nig_reg(y_true, gamma, v, alpha, beta)
    return loss_nll + coeff * loss_reg

if __name__ == "__main__":
    import functools
    import numpy as np
    import matplotlib.pyplot as plt

        #### Helper functions ####
    def my_data(x_min, x_max, n, train=True):
        x = torch.linspace(x_min, x_max, n)
        x = torch.unsqueeze(x, -1)

        sigma = 3 * torch.ones_like(x) if train else torch.zeros_like(x)
        y = x**3 + torch.normal(0, sigma)

        return x, y
    
    # Create some training and testing data
    x_train, y_train = my_data(-4, 4, 1000)
    x_test, y_test = my_data(-7, 7, 1000, train=False)

    # Define our model with an evidential output
    model = nn.Sequential(
        nn.LazyLinear(64), nn.ReLU(),
        nn.LazyLinear(64), nn.ReLU(),
        DenseNormalGamma(1),
    )
    opt = torch.optim.Adam(model.parameters())

    # Custom loss function to handle the custom regularizer coefficient
    def EvidentialRegressionLoss(true, pred):
        return evidential_regression(true, pred, coeff=1e-2)

    for epoch in range(500):
        idx = torch.randperm(1000).reshape(10, 100)
        for batch_idx in idx:
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]
            loss = EvidentialRegressionLoss(y_batch, model(x_batch)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch}, loss: {loss.item()}")

    def plot_predictions(x_train, y_train, x_test, y_test, y_pred, n_stds=4, kk=0):
        x_test = x_test[:, 0]
        mu, v, alpha, beta = torch.chunk(y_pred, 4, dim=-1)
        mu = mu[:, 0]
        var = np.sqrt(beta / (v * (alpha - 1)))
        var = np.minimum(var, 1e3)[:, 0]  # for visualization

        plt.figure(figsize=(5, 3), dpi=200)
        plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0, label="Train")
        plt.plot(x_test, y_test, 'r--', zorder=2, label="True")
        plt.plot(x_test, mu, color='#007cab', zorder=3, label="Pred")
        plt.plot([-4, -4], [-150, 150], 'k--', alpha=0.4, zorder=0)
        plt.plot([+4, +4], [-150, 150], 'k--', alpha=0.4, zorder=0)
        for k in np.linspace(0, n_stds, 4):
            plt.fill_between(
                x_test, (mu - k * var), (mu + k * var),
                alpha=0.3,
                edgecolor=None,
                facecolor='#00aeef',
                linewidth=0,
                zorder=1,
                label="Unc." if k == 0 else None)
        plt.gca().set_ylim(-150, 150)
        plt.gca().set_xlim(-7, 7)
        plt.legend(loc="upper left")
        plt.show()

    # Predict and plot using the trained model
    y_pred = model(x_test).detach()
    plot_predictions(x_train, y_train, x_test, y_test, y_pred)