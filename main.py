import torch
import torch.nn as nn
from torch import optim
from DepthDataset import DepthDataset, NyudDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import os
import visdom
from AE import Autoencoder
import numpy as np


def inf_data_gen(dataloader):
    while True:
        for data in dataloader:
            yield data


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight.data)
        nn.init.constant(m.bias.data, 0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal(m.weight.data)
        nn.init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0)


def nyud_test(vis, data_gen, autoencoder, prefix, env):
    autoencoder.eval()
    single_batch = next(data_gen)
    real_single_view = Variable(single_batch['real_single_view']).cuda()
    fake_all_views = autoencoder(real_single_view)
    single_batch = real_single_view.size()[0]
    for i in range(single_batch):
        vis.image(torch.unsqueeze(real_single_view.data[i], 1).cpu().numpy(),
                  opts=dict(title=prefix + "%d_real_single" % i), env=env)
        vis.images(torch.unsqueeze(fake_all_views.data[i], 1).cpu().numpy(), nrow=5,
                   opts=dict(title=prefix + "%d_fake_all" % i), env=env)


def visual(vis, real_all_views, real_single_view, autoencoder, prefix, env):
    fake_all_views = autoencoder(real_single_view)
    batch_size = real_single_view.size()[0]
    for i in range(batch_size):
        vis.image(torch.unsqueeze(real_single_view.data[i], 1).cpu().numpy(),
                  opts=dict(title=prefix + " %d_real_single" % i), env=env)
        vis.images(torch.unsqueeze(real_all_views.data[i], 1).cpu().numpy(), nrow=5,
                   opts=dict(title=prefix + " %d_real_all" % i), env=env)
        vis.images(torch.unsqueeze(fake_all_views.data[i], 1).cpu().numpy(), nrow=5,
                   opts=dict(title=prefix + " %d_fake_all" % i), env=env)
    vis.save([env])


def visual_single_batch(vis, data_gen, autoencoder, prefix, env):
    autoencoder.eval()
    single_batch = next(data_gen)
    real_all_views = Variable(single_batch['real_all_views']).cuda()
    real_single_view = Variable(single_batch['real_single_view']).cuda()
    visual(vis, real_all_views, real_single_view, autoencoder, prefix, env)


def train(train_dataloader, autoencoder, optimizer, model_dir, vis, category):
    print('Training the model')
    dataset = train_dataloader.dataset
    print('Training data size: %d' % len(dataset))
    # in case you need
    # image_size = 224 * 224
    num_iters = 100000
    data_gen = inf_data_gen(train_dataloader)
    env = category
    iter_loss = vis.line(
        X=np.zeros(1),
        Y=np.zeros(1),
        opts=dict(
            xlabel="iter",
            ylabel="loss",
            title="iter_loss",
            legend=["iter_loss"],
        ),
        env=env
    )
    iter_IoU = vis.line(
        X=np.zeros(1),
        Y=np.zeros(1),
        opts=dict(
            xlabel="iter",
            ylabel="IoU",
            title="iter_IoU",
            legend=["iter_IoU"],
        ),
        env = env
    )
    iter_mse = vis.line(
        X=np.zeros(1),
        Y=np.zeros(1),
        opts=dict(
            xlabel="iter",
            ylabel="mse",
            title="iter_mse",
            legend=["iter_mse"],
        ),
        env=env
    )
    vis.save([env])
    for i in range(1, num_iters + 1):
        # print("iter %d" % i)
        autoencoder.train()
        data = next(data_gen)
        real_all_views = Variable(data['real_all_views']).cuda()
        real_single_view = Variable(data['real_single_view']).cuda()
        optimizer.zero_grad()
        fake_all_views = autoencoder(real_single_view)
        ae_loss = nn.L1Loss()(fake_all_views, real_all_views)
        loss = ae_loss
        # print("ae_loss: %f" % ae_loss.data[0])
        vis.line(
            X=np.array([i]),
            Y=np.array([ae_loss.data[0]]),
            win=iter_loss,
            update="append",
            env=env
        )
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            # visual_single_batch(vis, data_gen, autoencoder, "iter %d" % i, category)
            mse, IoU = test(train_dataloader, autoencoder)
            vis.line(
                X=np.array([i]),
                Y=np.array([IoU]),
                win=iter_IoU,
                update="append",
                env=env
            )
            vis.line(
                X=np.array([i]),
                Y=np.array([mse]),
                win=iter_mse,
                update="append",
                env=env
            )
            vis.save([env])
            state_dicts = dict(autoencoder=autoencoder.state_dict())
            torch.save(state_dicts, os.path.join(model_dir, "ae_res_l1_%d.pkl" % i))


def calc_IU(real_all_views, fake_all_views):
    real_object = (real_all_views > 0.01).data
    fake_object = (fake_all_views > 0.01).data
    return torch.sum(real_object & fake_object), torch.sum(real_object | fake_object)


def test(test_dataloader, autoencoder):
    print('Testing the model')
    dataset = test_dataloader.dataset
    print('Testing data size: %d' % len(dataset))
    view_points = 20
    image_size = 224 * 224
    intersections = 0.0
    unions = 0.0
    # num_iters = 50
    # data_gan = inf_data_gen(test_dataloader)
    ave_mse = 0.0
    ave_IoU = 0.0
    autoencoder.eval()
    for data in test_dataloader:
        # data = next(data_gan)
        real_all_views = Variable(data['real_all_views']).cuda()
        real_single_view = Variable(data['real_single_view']).cuda()
        fake_all_views = autoencoder(real_single_view)
        intersection, union = calc_IU(real_all_views, fake_all_views)
        intersections += intersection
        unions += union
        IoU = intersection / union
        ave_IoU += IoU
        mse = nn.MSELoss(size_average=False)(fake_all_views, real_all_views)
        ave_mse += mse.data[0]
        del real_all_views, real_single_view, fake_all_views, mse
    ave_mse /= len(dataset) * view_points * image_size
    # IoU = intersections / unions
    ave_IoU /= len(test_dataloader)
    return ave_mse, ave_IoU


def main(args):
    print('Starting main procedure')
    # arg parse
    index_dir = args.index_dir
    model = args.model
    learning_rate = args.learning_rate
    mode = args.mode
    category = args.category

    # initialize
    model_dir = os.path.join(os.getcwd(), 'models', 'ae', category)
    os.makedirs(model_dir, exist_ok=True)
    vis = visdom.Visdom()
    batch_size = 8

    # ae-network
    autoencoder = Autoencoder().cuda()
    autoencoder.apply(init_weights)
    if model is not None:
        state_dicts = torch.load(model)
        autoencoder.load_state_dict(state_dicts['autoencoder'])

    # optimizer
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # dataset
    train_index = os.path.join(index_dir, "train_index.json")
    test_index = os.path.join(index_dir, "test_index.json")
    train_dataset = DepthDataset(train_index, category)
    test_dataset = DepthDataset(test_index, category)

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=3)

    if mode == 'training':
        train(train_dataloader, autoencoder, optimizer, model_dir, vis, category)
    elif mode == 'visualization':
        assert model is not None, "Visualization without specifying a model"
        visual_single_batch(vis, inf_data_gen(test_dataloader), autoencoder, "Visual", category)
    elif mode == 'testing':
        assert model is not None, "Testing without specifying a model"
        # visual_single_batch(vis, inf_data_gen(test_dataloader), autoencoder, "Test", category)
        mse, IoU = test(test_dataloader, autoencoder)
        print("mse: %f, IoU: %f" % (mse, IoU))
    elif mode == 'nyud':
        assert model is not None, "Testing nyud data without specifying a model"
        nyud_index = os.path.join(index_dir, "nyud_index.json")
        nyud_dataset = NyudDataset(nyud_index, category)
        nyud_dataloader = DataLoader(nyud_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
        nyud_test(vis, inf_data_gen(nyud_dataloader), autoencoder, "Nyud", category)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index_dir",
        help="index dir",
        default='/home/fz/Downloads/nonbenchmark/index'
    )
    parser.add_argument(
        "--model",
        help="the path of saved model",
        default=None
    )
    parser.add_argument(
        "--learning_rate",
        help="initial learning rate",
        type=float,
        default=0.0001
    )
    parser.add_argument(
        "--mode",
        help="choosing a mode from {training, visualization, testing, nyud}",
        choices=["training", "visualization", "testing", "nyud"],
        default="training"
    )
    parser.add_argument(
        "--category",
        help="which kind of data used in training",
        default="car"
    )
    args = parser.parse_args()
    main(args)
