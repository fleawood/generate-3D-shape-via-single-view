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
import json
import os
import visdom


def get_one_hot(y, num_classes):
    y = y.view(-1, 1)
    one_hot = Variable(torch.zeros(y.size()[0], num_classes)).cuda()
    one_hot.scatter_(1, y, 1)
    return one_hot


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


def visual(vis, all_data, single_data, label, vae, num_cats, prefix):
    depth_maps, mean, var, category = vae(single_data, get_one_hot(label, num_cats))
    batch_size = single_data.size()[0]
    for i in range(batch_size):
        vis.images(torch.unsqueeze(all_data.data[i], 1).cpu().numpy(), nrow=5,
                   opts=dict(title=prefix + " %d_real" % i))
        vis.images(torch.unsqueeze(depth_maps.data[i], 1).cpu().numpy(), nrow=5,
                   opts=dict(title=prefix + " %d_fake" % i))


def visual_single_batch(vis, valid_dataloader, vae, num_cats, prefix):
    single_batch = next(iter(valid_dataloader))
    all_data = Variable(single_batch['all_data']).cuda()
    single_data = Variable(single_batch['single_data']).cuda()
    label = Variable(single_batch['label']).cuda()
    visual(vis, all_data, single_data, label, vae, num_cats, prefix)


def evaluate(valid_dataloader, vae, depth_maps_loss, KLD_loss, classification_loss, num_cats, vis, prefix):
    print('Evaluating the model')
    print('Validation data size: %d' % len(valid_dataloader.dataset))
    vae.eval()
    image_size = 224 * 224
    num_views = 20
    r_loss = 0.0
    k_loss = 0.0
    c_loss = 0.0
    for _, sample_batched in enumerate(valid_dataloader):
        all_data = Variable(sample_batched['all_data']).cuda()
        single_data = Variable(sample_batched['single_data']).cuda()
        label = Variable(sample_batched['label']).cuda()
        depth_maps, mean, var, category = vae(single_data, get_one_hot(label, num_cats))
        r_loss += depth_maps_loss(depth_maps, all_data).data[0]
        k_loss += KLD_loss(mean, var).data[0]
        c_loss += classification_loss(category, label).data[0]
        # It is important to free the graph in evaluation time, or you will get
        # an "out of memory" error.
        # See https://github.com/pytorch/pytorch/issues/4050
        del all_data, label, depth_maps, mean, var, category
    r_loss /= len(valid_dataloader.dataset)
    k_loss /= len(valid_dataloader.dataset)
    c_loss /= len(valid_dataloader.dataset)
    loss = r_loss + KLD_loss.coeff * k_loss + c_loss
    print('Reconstruction loss: %f, KLD loss: %f, classification loss: %f' % (
        r_loss / (num_views * image_size), k_loss, c_loss))
    visual_single_batch(vis, valid_dataloader, vae, num_cats, prefix)
    return loss


def train(train_dataloader, valid_dataloader, model_epochs, num_epochs, vae, depth_maps_loss, KLD_loss,
          classification_loss, scheduler, num_cats, model_dir, vis):
    print('Training the model')
    print('Training data size: %d' % len(train_dataloader.dataset))
    # num_epochs = 10
    optimizer = scheduler.optimizer
    # in case you need
    # image_size = 224 * 224
    for i in range(model_epochs + 1, model_epochs + num_epochs + 1):
        print('Starting epoch %d' % i)
        vae.train()
        for _, sample_batched in enumerate(train_dataloader):
            all_data = Variable(sample_batched['all_data']).cuda()
            single_data = Variable(sample_batched['single_data']).cuda()
            label = Variable(sample_batched['label']).cuda()
            optimizer.zero_grad()
            depth_maps, mean, var, category = vae(single_data, get_one_hot(label, num_cats))
            batch_size = all_data.size()[0]
            d_loss = depth_maps_loss(depth_maps, all_data) / batch_size
            k_loss = KLD_loss(mean, var) / batch_size
            c_loss = classification_loss(category, label) / batch_size
            loss = d_loss + KLD_loss.coeff * k_loss + c_loss
            loss.backward()
            optimizer.step()
        print('Finished epoch %d' % i)
        torch.save(vae.state_dict(), os.path.join(model_dir, "vae_%d.pkl" % i))

        print('Evaluating result using validation data')
        loss = evaluate(valid_dataloader, vae, depth_maps_loss, KLD_loss, classification_loss, num_cats, vis,
                        "Epoch %d" % i)
        scheduler.step(loss)


def test(test_dataloader, vae, depth_maps_loss, KLD_loss, classification_loss, num_cats, vis):
    print('Testing the model')
    print('Testing data size: %d', len(test_dataloader.dataset))
    # TODO: some test


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

    # initialize
    model_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(model_dir, exist_ok=True)
    labels_filename = os.path.join(index_dir, "labels.json")
    with open(labels_filename, "r") as file:
        labels = json.load(file)
    num_cats = len(labels)
    vis = visdom.Visdom()
    batch_size = 16

    # vae network
    vae = VAE(num_cats).cuda()
    vae.apply(init_weights)
    if model is not None:
        vae.load_state_dict(torch.load(model))

    # criterion
    classification_loss = nn.CrossEntropyLoss(size_average=False)
    depth_maps_loss = nn.L1Loss(size_average=False)
    KLD_loss = KLDCriterion(KLD_coeff)

    # optimizer
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1)

    # dataset
    train_index = os.path.join(index_dir, "train_index.json")
    valid_index = os.path.join(index_dir, "valid_index.json")
    test_index = os.path.join(index_dir, "test_index.json")
    train_dataset = DepthDataset(train_index)
    valid_dataset = DepthDataset(valid_index)
    test_dataset = DepthDataset(test_index)

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if mode == 'training':
        train(train_dataloader, valid_dataloader, model_epochs, num_epochs, vae, depth_maps_loss, KLD_loss,
              classification_loss,
              scheduler, num_cats, model_dir, vis)
    elif mode == 'evaluation':
        assert model is not None, 'Evaluating without specifying a model'
        evaluate(valid_dataloader, vae, depth_maps_loss, KLD_loss, classification_loss, num_cats, vis, "Eval")
    elif mode == 'visualization':
        assert model is not None, 'Visualization without specifying a model'
        visual_single_batch(vis, valid_dataloader, vae, num_cats, "Visual")
        # test(test_dataloader, vae, depth_maps_loss, KLD_loss, classification_loss, num_cats, vis)


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
        default=10
    )
    parser.add_argument(
        "--KLD_coeff",
        help="coefficient of KLD Loss",
        type=float,
        default=100.0
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
        choices=['training', 'evaluation', 'visualization', 'testing'],
        default='training'
    )
    args = parser.parse_args()
    main(args)
