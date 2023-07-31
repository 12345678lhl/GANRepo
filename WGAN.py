import torch
from torch import nn, optim, autograd
import pandas as pd
import numpy as np
# import visdom
import random
from matplotlib import pyplot as plt

h_dim = 30
batch_size = 30

# nor_label = np.random.randint(45000)
nor_label = 3000
nor_data = pd.read_csv("After_Thompson_Tau_And_Quartile.csv", usecols=[1, 2])

radiation_true = nor_data.iloc[nor_label:nor_label+batch_size:, 1]
power_true = nor_data.iloc[nor_label:nor_label+batch_size:, 0]

nor_data = np.array(nor_data.iloc[nor_label:nor_label+batch_size, :])
print(nor_data)

abnor_label = np.random.randint(1000)
abnormal_data = pd.read_csv("abnormal_data.csv", usecols=[1, 2])
abnormal_data = np.array(abnormal_data.iloc[abnor_label:abnor_label+batch_size, :])
abnormal_data = torch.from_numpy(abnormal_data)
abnormal_data = abnormal_data.to(torch.float32)


# viz = visdom.Visdom()


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(

            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2),
            nn.Tanh()
        )

    def forward(self, z):  # z为隐藏变量
        output = self.net(z)
        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(

            nn.Linear(2, h_dim),
            nn.LeakyReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()

        )

    def forward(self, x):  # z为隐藏变量
        output = self.net(x)
        return output.view(-1)


def data_generator():

    # scale = 2.
    # centers = nor_data
    # centers = [(scale * x, scale * y) for x, y in centers]

    while True:
        dataset = []
        for i in range(batch_size):
            point = np.random.randn(2) * 0.02
            # center = random.choice(centers)
            center = nor_data[i]
            # N(0, 1) + center_x1/x2
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
            # dataset.append(center)
        dataset = np.array(dataset).astype(np.float32)
        # dataset /= 1.414
        yield dataset


def generator_image(D, G, xr, epoch):

    N_POINTS = 128
    RANGE = 3
    plt.clf()

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))
    # (16384, 2)
    # print('p:', points.shape)

    # draw contour
    with torch.no_grad():
        points = torch.Tensor(points).cuda()  # [16384, 2]
        disc_map = D(points).cpu().numpy()  # [16384]
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.clabel(cs, inline=1, fontsize=10)
    # plt.colorbar()

    # draw samples
    with torch.no_grad():
        z = torch.randn(batch_size, 2).cuda()  # [b, 2]
        samples = G(z).cpu().numpy()  # [b, 2]
    plt.scatter(xr[:, 0], xr[:, 1], c='orange', marker='.')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    # viz.matplot(plt, win='contour', opts=dict(title='p(x):%d' % epoch))


def gradient_penalty(D, x_real, x_abnormal):
    # [b, 1]
    t = torch.rand(batch_size, 1).cuda()
    # [b, 1] => [b, 2]
    t = t.expand_as(x_real)
    # 在x_real与x_abnormal之间做一个线性插值
    mid = t * x_real + (1 - t) * x_abnormal
    # 设置它需要的导数信息
    mid.requires_grad_()
    pred = D(mid)
    grads = autograd.grad(outputs=pred, inputs=mid,
                          grad_outputs=torch.ones_like(pred),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()
    return gp


def main():
    torch.manual_seed(23)
    np.random.seed(23)

    data_iter = data_generator()
    x = next(data_iter)  # x:[b, 2]
    print(x.shape)

    G = Generator().cuda()
    D = Discriminator().cuda()

    print(G)
    print(D)

    optim_G = optim.Adam(G.parameters(), lr=5e-4, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=5e-4, betas=(0.5, 0.9))

    loss_fn = torch.nn.BCELoss()

    # viz.line([[0, 0]], [0], win="loss", opts=dict(title="loss", legend=["D", "G"]))

    for epoch in range(5000):
        #  首先训练 Discriminator
        for _ in range(5):
            # 1、训练真实数据
            x_real = next(data_iter)
            x_real = torch.from_numpy(x_real).cuda()
            pred_real = D(x_real)
            # 使得predr最大, lossr最小

            # loss_real = -pred_real.mean()
            loss_real = loss_fn(pred_real, torch.ones_like(pred_real))

            # 2、训练异常数据
            # z = torch.randn(batch_size, 2).cuda()
            # z = torch.from_numpy(abnormal_data).cuda()
            z = abnormal_data.cuda()
            x_abnormal = G(z).detach()  # tf.stop.gradient()
            pred_abnormal = D(x_abnormal)

            # loss_abnormal = pred_abnormal.mean()
            loss_abnormal = loss_fn(pred_abnormal, torch.zeros_like(pred_abnormal))

            # 3、梯度惩罚
            gp = gradient_penalty(D, x_real, x_abnormal.detach())

            # aggregate all
            # loss_D = loss_real + loss_abnormal
            loss_D = loss_real + loss_abnormal + 0.2 * gp

            # 优化Discriminator模型
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        # 训练 Generator
        # z = torch.randn(batch_size, 2).cuda()
        # z = torch.from_numpy(abnormal_data).cuda()
        z = abnormal_data.cuda()
        x_abnormal = G(z)
        pred_abnormal = D(x_abnormal)
        # 使得pred_abnormal.mean()最大
        # loss_G = -pred_abnormal.mean()
        loss_G = loss_fn(pred_abnormal, torch.ones_like(pred_abnormal))

        # print(type(x_abnormal))
        # a = x_abnormal.detach().cpu().numpy()
        # a = pd.DataFrame(a)
        # print(type(a))
        # a.to_csv("x_abnormal.csv", index=None)
        # print(x_abnormal)
        # print(x_abnormal.shape)
        # print("-------------")

        # 优化Generator模型
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        # if epoch % 100 == 0:
        #     # viz.line([[loss_D.item(), loss_G.item()]], [epoch], win="loss", update="append")
        #     print(loss_D.item(), loss_G.item())
            # generator_image(D, G, x_real.cpu(), epoch)

    a = x_abnormal.detach().cpu().numpy()
    a = pd.DataFrame(a)

    a.to_csv("x_abnormal.csv", index=None)
    print("生成器的损失：", loss_G.item())
    print("判别器的损失：", loss_D.item())


if __name__ == "__main__":
    main()
