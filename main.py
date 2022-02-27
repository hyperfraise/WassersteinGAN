from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
import json
import numpy as np
from torch.autograd import Variable

import models.dcgan as dcgan
import models.mlp as mlp


def select_images_random_patches(images, patchSize):
    patches = []
    for image in images:
        x, y = np.random.choice(
            image.size(-2)-patchSize), np.random.choice(image.size(-2)-patchSize)
        patch = image[:, x:x+patchSize, y:y+patchSize]
        patches.append(patch)
    return torch.stack(patches)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        help='cifar10 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int,
                        default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=256,
                        help='the height / width of the input image to network')
    parser.add_argument('--patchSize', type=int, default=64,
                        help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=3,
                        help='input image channels')
    parser.add_argument('--nz', type=int, default=100,
                        help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25,
                        help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.00005,
                        help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005,
                        help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='number of GPUs to use')
    parser.add_argument('--netG', default='',
                        help="path to netG (to continue training)")
    parser.add_argument('--netD', default='',
                        help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=5,
                        help='number of D iters per each G iter')
    parser.add_argument('--start-iter', type=int, default=0,
                        help='number of D iters per each G iter')

    parser.add_argument('--start-gen-iterations', type=int, default=0,
                        help='number of D iters per each G iter')

    parser.add_argument('--noBN', action='store_true',
                        help='use batchnorm or not (only for DCGAN)')
    parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
    parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
    parser.add_argument('--n_extra_layers', type=int, default=0,
                        help='Number of extra layers on gen and disc')
    parser.add_argument('--experiment', default=None,
                        help='Where to store samples and models')
    parser.add_argument('--adam', action='store_true',
                        help='Whether to use adam (default is rmsprop)')
    opt = parser.parse_args()
    print(opt)

    if opt.experiment is None:
        opt.experiment = 'samples'
    os.system('mkdir {0}'.format(opt.experiment))

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                    transforms.Scale(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Scale(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                               transform=transforms.Compose([
                                transforms.Scale(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ])
                               )
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = int(opt.nc)
    n_extra_layers = int(opt.n_extra_layers)

    # write out generator config to generate images together wth training checkpoints (.pth)
    generator_config = {"imageSize": opt.imageSize, "nz": nz, "nc": nc, "ngf": ngf,
                        "ngpu": ngpu, "n_extra_layers": n_extra_layers, "noBN": opt.noBN, "mlp_G": opt.mlp_G}
    with open(os.path.join(opt.experiment, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config)+"\n")

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    if opt.noBN:
        netG = dcgan.DCGAN_G_nobn(
            opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    elif opt.mlp_G:
        netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
    else:
        netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

    # write out generator config to generate images together wth training checkpoints (.pth)
    generator_config = {"imageSize": opt.imageSize, "nz": nz, "nc": nc, "ngf": ngf,
                        "ngpu": ngpu, "n_extra_layers": n_extra_layers, "noBN": opt.noBN, "mlp_G": opt.mlp_G}
    with open(os.path.join(opt.experiment, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config)+"\n")

    netG.apply(weights_init)
    if opt.netG != '':  # load checkpoint if needed
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    if opt.mlp_D:
        netD = mlp.MLP_D(opt.patchSize, nz, nc, ndf, ngpu)
    else:
        netD = dcgan.DCGAN_D(opt.patchSize, nz, nc, ndf, ngpu, n_extra_layers)
        netD.apply(weights_init)

    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    input = torch.FloatTensor(opt.batchSize, 3, opt.patchSize, opt.patchSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    one = torch.FloatTensor([1])
    mone = one * -1

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        input = input.cuda()
        one, mone = one.cuda(), mone.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    # setup optimizer
    if opt.adam:
        optimizerD = optim.Adam(
            netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(
            netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)

    fixed_input_criterion = nn.MSELoss().cuda()
    siamese_criterion = nn.MSELoss().cuda()

    gen_iterations = opt.start_gen_iterations
    i = opt.start_iter

    data_iter = iter(dataloader)
    real_images_batch, _ = data_iter.next()
    # train with real
    if opt.cuda:
        real_images_batch = real_images_batch.cuda()
    batch_size = real_images_batch.size(0)

    fixed_noise.resize_(2, nz, 1, 1).normal_(0, 1)
    fixed_noisev = Variable(fixed_noise)

    while 1:
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 100 == 0:
            Diters = 25
        else:
            Diters = opt.Diters
        j = 0
        while j < Diters:
            j += 1

            # clamp parameters to a cube
            for p in netD.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            i += 1
            netD.zero_grad()
            real_patches = select_images_random_patches(
                real_images_batch, opt.patchSize)

            input.resize_as_(real_patches).copy_(real_patches)
            inputv = Variable(input)
            errD_real, embedding = netD(inputv)
            errD_real.backward(one)

            # train with fake
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            with torch.no_grad():
                fake = Variable(netG(noisev))
            fake_patches = select_images_random_patches(
                fake.data, opt.patchSize)
            inputv = Variable(fake_patches)
            errD_fake, embedding = netD(inputv)
            errD_fake.backward(mone)
            errD = errD_real - errD_fake
            optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        fake_patches = select_images_random_patches(
            fake, opt.patchSize)
        errG, embedding = netD(fake_patches)
        errG.backward(one)

        ############################
        # (3) Make the generator predict the images
        ###########################
        fixed_fake = netG(fixed_noisev)
        fixed_input_loss = fixed_input_criterion(fixed_fake, real_images_batch)
        fixed_input_loss.backward()
        gen_iterations += 1

        ############################
        # (4) Smooth the discriminator embeddings of the generator outputs
        ###########################
        noise.resize_(2, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        embedding_fake = netG(noisev)
        embedding_fake_patches = select_images_random_patches(
            embedding_fake, opt.patchSize)
        _, embedding = netD(embedding_fake_patches)
        siamese_loss = 0.005 * siamese_criterion(torch.linalg.norm(
            embedding[1] - embedding[0], ord=2), torch.linalg.norm((noise[1] - noise[0]).squeeze(-1).squeeze(-1), ord=2))
        siamese_loss.backward()
        optimizerG.step()

        print('[%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f Loss_G_Fixed %f Loss_G_embdedding %f'
              % (i, gen_iterations,
                 errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0], fixed_input_loss.data, siamese_loss.data))
        if gen_iterations % 50 == 0:
            real_cpu = real_images_batch[0].mul(0.5).add(0.5)
            vutils.save_image(
                real_cpu, '{0}/real_samples.png'.format(opt.experiment))
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            with torch.no_grad():
                fake = netG(noise)
            fake.data = fake.data.mul(0.5).add(0.5)
            vutils.save_image(
                fake.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))
            step = 1/opt.batchSize
            vis_noise = torch.stack(
                [fixed_noise[0]*(1-t) + fixed_noise[1]*t for t in np.arange(0, 1+step, step)])
            with torch.no_grad():
                vis_fake = netG(vis_noise)
            fake.data = fake.data.mul(0.5).add(0.5)
            # vutils.save_image(
            #     fake.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))

        if i % 1000 == 0:
            # do checkpointing
            torch.save(netG.state_dict(),
                       '{0}/netG_iter_{1}.pth'.format(opt.experiment, i))
            torch.save(netD.state_dict(),
                       '{0}/netD_iter_{1}.pth'.format(opt.experiment, i))
