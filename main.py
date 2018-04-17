import torch
import torch.nn as nn
import torch.optim as optim
from VAE import VAE
from DepthDataset import DepthDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from KLDCriterion import KLDCriterion
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os
import visdom


# def get_one_hot(y, num_classes):
#     y = y.view(-1, 1)
#     one_hot = Variable(torch.zeros(y.size()[0], num_classes)).cuda()
#     one_hot.scatter_(1, y, 1)
#     return one_hot


def init_weights(m):
    # print(m)
    if isinstance(m, nn.Linear):
        nn.init.normal(m.weight.data, 0, 0.02)
        nn.init.constant(m.bias.data, 0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal(m.weight.data)
        nn.init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0)


def visual(vis, real_all_views, real_single_view, vae, prefix, env):
    fake_all_views, mean, var = vae(real_single_view)
    batch_size = real_single_view.size()[0]
    for i in range(batch_size):
        vis.image(torch.unsqueeze(real_single_view.data[i], 1).cpu().numpy(),
                  opts=dict(title=prefix + " %d_real_single" % i), env=env)
        vis.images(torch.unsqueeze(real_all_views.data[i], 1).cpu().numpy(), nrow=5,
                   opts=dict(title=prefix + " %d_real_all" % i), env=env)
        vis.images(torch.unsqueeze(fake_all_views.data[i], 1).cpu().numpy(), nrow=5,
                   opts=dict(title=prefix + " %d_fake_all" % i), env=env)
    vis.save([env])


def visual_single_batch(vis, valid_dataloader, vae, prefix, env):
    single_batch = next(iter(valid_dataloader))
    real_all_views = Variable(single_batch['real_all_views']).cuda()
    real_single_view = Variable(single_batch['real_single_view']).cuda()
    visual(vis, real_all_views, real_single_view, vae, prefix, env)


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


def train(train_dataloader, valid_dataloader, model_epochs, num_epochs, vae, reconstruction_loss, KLD_loss, scheduler,
          model_dir, vis, category):
    print('Training the model')
    dataset = train_dataloader.dataset
    print('Training data size: %d' % len(dataset))
    optimizer = scheduler.optimizer
    # in case you need
    # image_size = 224 * 224
    for i in range(model_epochs + 1, model_epochs + num_epochs + 1):
        print('Starting epoch %d' % i)
        vae.train()
        ave_loss = 0.0
        ave_r_loss = 0.0
        ave_k_loss = 0.0
        for _, sample_batched in enumerate(train_dataloader):
            real_all_views = Variable(sample_batched['real_all_views']).cuda()
            real_single_view = Variable(sample_batched['real_single_view']).cuda()
            optimizer.zero_grad()
            fake_all_views, mean, var = vae(real_single_view)
            r_loss = reconstruction_loss(fake_all_views, real_all_views)
            k_loss = KLD_loss(mean, var)
            loss = r_loss + KLD_loss.coeff * k_loss
            ave_r_loss += r_loss.data[0]
            ave_k_loss += k_loss.data[0]
            ave_loss += loss.data[0]
            loss.backward()
            optimizer.step()
        ave_r_loss /= len(dataset)
        ave_k_loss /= len(dataset)
        ave_loss /= len(dataset)
        print("loss: %f, Reconstruction loss: %f, KLD loss: %f" % (ave_loss, ave_r_loss, ave_k_loss))
        scheduler.step(ave_loss)
        torch.save(vae.state_dict(), os.path.join(model_dir, "vae_%d.pkl" % i))
        test(train_dataloader, vae)
        print('Finished epoch %d' % i)
        if i % 5 == 0:
            print("Evaluating result using validation dataset")
            evaluate(valid_dataloader, vae, reconstruction_loss, KLD_loss, vis, "Epoch %d evaluation" % i,
                     category + " " + str(i))
            test(valid_dataloader, vae)


def calc_IU(real_all_views, fake_all_views):
    real_object = (real_all_views > 0.01).data
    fake_object = (fake_all_views > 0.01).data
    return torch.sum(real_object & fake_object), torch.sum(real_object | fake_object)


def test(test_dataloader, vae):
    print('Testing the model')
    dataset = test_dataloader.dataset
    print('Testing data size: %d' % len(dataset))
    intersections = 0.0
    unions = 0.0
    for _, sample_batched in enumerate(test_dataloader):
        real_all_views = Variable(sample_batched['real_all_views']).cuda()
        real_single_view = Variable(sample_batched['real_single_view']).cuda()
        fake_all_views, mean, var = vae(real_single_view)
        intersection, union = calc_IU(real_all_views, fake_all_views)
        intersections += intersection
        unions += union
        del real_all_views, real_single_view, fake_all_views, mean, var
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
    model_dir = os.path.join(os.getcwd(), 'models', category)
    os.makedirs(model_dir, exist_ok=True)
    vis = visdom.Visdom()
    batch_size = 8

    # vae network
    vae = VAE().cuda()
    vae.apply(init_weights)
    if model is not None:
        vae.load_state_dict(torch.load(model))

    # criterion
    reconstruction_loss = nn.MSELoss(size_average=False)
    KLD_loss = KLDCriterion(KLD_coeff)

    # optimizer
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4)

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
        train(train_dataloader, valid_dataloader, model_epochs, num_epochs, vae, reconstruction_loss, KLD_loss,
              scheduler, model_dir, vis, category)
    elif mode == 'evaluation':
        assert model is not None, "Evaluating without specifying a model"
        evaluate(valid_dataloader, vae, reconstruction_loss, KLD_loss, vis, "Eval", category)
    elif mode == 'visualization':
        assert model is not None, "Visualization without specifying a model"
        visual_single_batch(vis, valid_dataloader, vae, "Visual", category)
    elif mode == 'testing':
        assert model is not None, "Testing without specifying a model"
        test(test_dataloader, vae)


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
        help="choosing a mode from {training, evaluation, visualization, testing}",
        choices=["training", "evaluation", "visualization", "testing"],
        default="training"
    )
    parser.add_argument(
        "--category",
        help="which kind of data used in training",
        default="lamp"
    )
    args = parser.parse_args()
    main(args)
