import torch
import torch.nn as nn
from torch import optim
# from VAE import VAE
from DepthDataset import DepthDataset, NyudDataset
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
# from KLDCriterion import KLDCriterion
# from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os
import visdom
from AEGAN import Autoencoder, Discriminator
import numpy as np


# def get_one_hot(y, num_classes):
#     y = y.view(-1, 1)
#     one_hot = Variable(torch.zeros(y.size()[0], num_classes)).cuda()
#     one_hot.scatter_(1, y, 1)
#     return one_hot


def inf_data_gen(dataloader):
    while True:
        for data in dataloader:
            yield data


def init_weights(m):
    # print(m)
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
    # real_single_view.data[:, :, :80].zero_()
    # visual(vis, real_all_views, real_single_view, autoencoder, prefix, env)


def evaluate(valid_dataloader, vae, reconstruction_loss, KLD_loss, vis, prefix, env):
    print('Evaluating the model')
    dataset = valid_dataloader.dataset
    print('Validation data size: %d' % len(dataset))
    vae.eval()
    # image_size = 224 * 224
    # num_views = 20
    r_loss = 0.0
    k_loss = 0.0
    for _, sample_batched in enumerate(valid_dataloader):
        real_all_views = Variable(sample_batched['real_all_views']).cuda()
        real_single_view = Variable(sample_batched['real_single_view']).cuda()
        fake_all_views, mean, var = vae(real_single_view)
        r_loss += reconstruction_loss(fake_all_views, real_all_views).data[0]
        k_loss += KLD_loss(mean, var).data[0]
        # It is important to free the graph in evaluation time, or you will get
        # an "out of memory" error.
        # See https://github.com/pytorch/pytorch/issues/4050
        del real_all_views, real_single_view, fake_all_views, mean, var
    r_loss /= len(dataset)
    k_loss /= len(dataset)
    loss = r_loss + KLD_loss.coeff * k_loss
    print("loss: %f, Reconstruction loss: %f, KLD loss: %f" % (loss, r_loss, k_loss))
    visual_single_batch(vis, valid_dataloader, vae, prefix, env)


def gradient_penalty(discriminator, fake_all_views, real_all_views):
    batch_size = real_all_views.size()[0]
    alpha = torch.rand(batch_size, 1, 1, 1).cuda()
    interpolates = Variable(alpha * real_all_views.data + (1 - alpha) * fake_all_views.data, requires_grad=True)
    d_interpolates = discriminator(interpolates)
    gradients = grad(d_interpolates, interpolates, torch.ones(d_interpolates.size()).cuda(), create_graph=True)[0]
    gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp


def train(train_dataloader, valid_dataloader, autoencoder, discriminator, g_optimizer, d_optimizer,
          model_dir, vis, category):
    print('Training the model')
    dataset = train_dataloader.dataset
    print('Training data size: %d' % len(dataset))
    # in case you need
    # image_size = 224 * 224
    num_iters = 100
    num_critic_iters = 3
    num_gen_iters = 3
    data_gen = inf_data_gen(train_dataloader)
    for i in range(1, num_iters + 1):
        print("iter %d" % i)
        autoencoder.train()
        for p in discriminator.parameters():
            p.requires_grad = True
        for p in autoencoder.parameters():
            p.requires_grad = False
        for j in range(num_critic_iters):
            data = next(data_gen)
            real_all_views = Variable(data['real_all_views']).cuda()
            real_single_view = Variable(data['real_single_view']).cuda()
            d_optimizer.zero_grad()
            fake_all_views = autoencoder(real_single_view)
            d_real = discriminator(real_all_views)
            d_fake = discriminator(fake_all_views)
            gp = gradient_penalty(discriminator, fake_all_views, real_all_views)
            d_loss = torch.mean(d_fake) - torch.mean(d_real) + 10.0 * gp
            print("d_loss: %f, gp: %f" % (d_loss.data[0], gp.data[0]))
            loss = d_loss
            loss.backward()
            d_optimizer.step()
        for p in discriminator.parameters():
            p.requires_grad = False
        for p in autoencoder.parameters():
            p.requires_grad = True
        for j in range(num_gen_iters):
            data = next(data_gen)
            real_all_views = Variable(data['real_all_views']).cuda()
            real_single_view = Variable(data['real_single_view']).cuda()
            g_optimizer.zero_grad()
            fake_all_views = autoencoder(real_single_view)
            d_fake = discriminator(fake_all_views)
            ae_loss = nn.MSELoss()(fake_all_views, real_all_views)
            g_loss = -torch.mean(d_fake)
            if i <= 10000:
                alpha = 1
                beta = 0
            else:
                alpha = 1
                beta = 0
            # mu = np.power(10.0, np.log10(np.clip(np.abs(g_loss.data[0].numpy()), 1.0, None)))
            loss = alpha * ae_loss + beta * g_loss
            print("ae_loss: %f, g_loss: %f" % (ae_loss.data[0], g_loss.data[0]))
            loss.backward()
            g_optimizer.step()
        if i % 10 == 0:
            visual_single_batch(vis, data_gen, autoencoder, "iter %d" % i, category)
            state_dicts = dict(autoencoder=autoencoder.state_dict(), discriminator=discriminator.state_dict())
            # torch.save(state_dicts, os.path.join(model_dir, "aegan_%d.pkl" % i))
            test(train_dataloader, autoencoder)
            test(valid_dataloader, autoencoder)

            # for i in range(model_epochs + 1, model_epochs + num_epochs + 1):
            #     print('Starting epoch %d' % i)
            #     ave_loss = 0.0
            #     ave_r_loss = 0.0
            #     ave_k_loss = 0.0
            #
            #     for _, sample_batched in enumerate(train_dataloader):
            #         real_all_views = Variable(sample_batched['real_all_views']).cuda()
            #         real_single_view = Variable(sample_batched['real_single_view']).cuda()
            #         optimizer.zero_grad()
            #         fake_all_views, mean, var = vae(real_single_view)
            #         r_loss = reconstruction_loss(fake_all_views, real_all_views)
            #         k_loss = KLD_loss(mean, var)
            #         loss = r_loss + KLD_loss.coeff * k_loss
            #         ave_r_loss += r_loss.data[0]
            #         ave_k_loss += k_loss.data[0]
            #         ave_loss += loss.data[0]
            #         loss.backward()
            #         optimizer.step()
            #     ave_r_loss /= len(dataset)
            #     ave_k_loss /= len(dataset)
            #     ave_loss /= len(dataset)
            #     print("loss: %f, Reconstruction loss: %f, KLD loss: %f" % (ave_loss, ave_r_loss, ave_k_loss))
            #     scheduler.step(ave_loss)
            #     torch.save(vae.state_dict(), os.path.join(model_dir, "vae_%d.pkl" % i))
            #     test(train_dataloader, vae)
            #     print('Finished epoch %d' % i)
            #     if i % 5 == 0:
            #         print("Evaluating result using validation dataset")
            #         evaluate(valid_dataloader, vae, reconstruction_loss, KLD_loss, vis, "Epoch %d evaluation" % i,
            #                  category + " " + str(i))
            #         test(valid_dataloader, vae)


def calc_IU(real_all_views, fake_all_views):
    real_object = (real_all_views > 0.01).data
    fake_object = (fake_all_views > 0.01).data
    return torch.sum(real_object & fake_object), torch.sum(real_object | fake_object)


def test(test_dataloader, autoencoder):
    print('Testing the model')
    dataset = test_dataloader.dataset
    print('Testing data size: %d' % len(dataset))
    intersections = 0.0
    unions = 0.0
    num_iters = 10
    data_gan = inf_data_gen(test_dataloader)
    for i in range(num_iters):
        data = next(data_gan)
        real_all_views = Variable(data['real_all_views']).cuda()
        real_single_view = Variable(data['real_single_view']).cuda()
        fake_all_views = autoencoder(real_single_view)
        intersection, union = calc_IU(real_all_views, fake_all_views)
        intersections += intersection
        unions += union
        del real_all_views, real_single_view, fake_all_views
    print("IoU: %f" % (intersections / unions))


def main(args):
    print('Starting main procedure')
    # arg parse
    index_dir = args.index_dir
    model = args.model
    model_epochs = args.model_epochs
    num_epochs = args.num_epochs
    KLD_coeff = args.KLD_coeff
    learning_rate = args.learning_rate
    mode = args.mode
    category = args.category

    # initialize
    model_dir = os.path.join(os.getcwd(), 'models', 'aegan', category)
    os.makedirs(model_dir, exist_ok=True)
    vis = visdom.Visdom()
    batch_size = 4

    # vae network
    # vae = VAE().cuda()
    # vae.apply(init_weights)
    # if model is not None:
    #     vae.load_state_dict(torch.load(model))

    # ae-gan network
    autoencoder = Autoencoder().cuda()
    discriminator = Discriminator().cuda()
    autoencoder.apply(init_weights)
    discriminator.apply(init_weights)
    if model is not None:
        state_dicts = torch.load(model)
        autoencoder.load_state_dict(state_dicts['autoencoder'])
        discriminator.load_state_dict(state_dicts['discriminator'])

    # optimizer
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.00005)
    g_optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4)

    # dataset
    train_index = os.path.join(index_dir, "train_index.json")
    valid_index = os.path.join(index_dir, "valid_index.json")
    test_index = os.path.join(index_dir, "test_index.json")
    train_dataset = DepthDataset(train_index, category)
    valid_dataset = DepthDataset(valid_index, category)
    test_dataset = DepthDataset(test_index, category)

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=3)

    if mode == 'training':
        train(train_dataloader, valid_dataloader, autoencoder, discriminator,
              g_optimizer, d_optimizer, model_dir, vis, category)
    elif mode == 'evaluation':
        assert model is not None, "Evaluating without specifying a model"
        # evaluate(valid_dataloader, vae, reconstruction_loss, KLD_loss, vis, "Eval", category)
    elif mode == 'visualization':
        assert model is not None, "Visualization without specifying a model"
        visual_single_batch(vis, inf_data_gen(valid_dataloader), autoencoder, "Visual", category)
    elif mode == 'testing':
        assert model is not None, "Testing without specifying a model"
        # test(test_dataloader, autoencoder)
        visual_single_batch(vis, inf_data_gen(test_dataloader), autoencoder, "Test", category)
    elif mode == 'nyud':
        assert model is not None, "Testing nyud data without specifying a model"
        nyud_index = os.path.join(index_dir, "nyud_index.json")
        nyud_dataset = NyudDataset(nyud_index, category)
        nyud_dataloader = DataLoader(nyud_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
        nyud_test(vis, inf_data_gen(nyud_dataloader), autoencoder, "Nyud", category)


if __name__ == '__main__':
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
        "--model_epochs",
        help="how many epochs have been trained",
        type=int,
        default=0
    )
    parser.add_argument(
        "--num_epochs",
        help="how many epochs need to be trained",
        type=int,
        default=100
    )
    parser.add_argument(
        "--KLD_coeff",
        help="coefficient of KLD Loss",
        type=float,
        default=60.0
    )
    parser.add_argument(
        "--learning_rate",
        help="initial learning rate",
        type=float,
        default=0.0001
    )
    parser.add_argument(
        "--mode",
        help="choosing a mode from {training, evaluation, visualization, testing, nyud}",
        choices=["training", "evaluation", "visualization", "testing", "nyud"],
        default="training"
    )
    parser.add_argument(
        "--category",
        help="which kind of data used in training",
        default="chair"
    )
    args = parser.parse_args()
    main(args)
