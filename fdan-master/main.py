import math
import random

random.seed(1)
import numpy as np

np.random.seed(1)

import argparse
# from model.digit5 import CNN, Classifier
# from model.amazon import AmazonMLP, AmazonClassifier
# from model.officecaltech10 import OfficeCaltechNet, OfficeCaltechClassifier
# from model.domainnet import DomainNet, DomainNetClassifier
# from lib.utils.federated_utils import *
# from train.train import train, test
from model.model import FNet, FNetClassifier_s, FNetClassifier_t
import os
from os import path
import shutil
import yaml
import ipdb
from dataload import dataloader

# Default settings
parser = argparse.ArgumentParser(description='FDAN')
# Dataset Parameters
parser.add_argument("--config", default="DigitFive.yaml")
parser.add_argument('-bp', '--base-path', default="/home/daifengqi/gitrep/fdan-master/data")
parser.add_argument('--target-domain', type=str, help="The target domain we want to perform domain adaptation")
parser.add_argument('--source-domains', type=str, nargs="+", help="The source domains we want to use")
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
# Train Strategy Parameters
parser.add_argument('-t', '--train-time', default=1, type=str,
                    metavar='N', help='the x-th time of training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-dp', '--data-parallel', action='store_false', help='Use Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# Optimizer Parameters
parser.add_argument('--optimizer', default="SGD", type=str, metavar="Optimizer Name")
parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M', help='Momentum in SGD')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float)
parser.add_argument('-bm', '--bn-momentum', type=float, default=0.1, help="the batchnorm momentum parameter")
parser.add_argument("--gpu", default="1,2,3,4", type=str, metavar='GPU plans to use', help='The GPU id plans to use')
args = parser.parse_args()
# import config files
with open(r"./config/{}".format(args.config)) as file:
    configs = yaml.full_load(file)
# set the visible GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

torch.manual_seed(1)
torch.cuda.manual_seed(1)


def main(args=args, configs=configs):
    #data prepareing
    train_dloaders, test_dloaders, num_classes = dataloader(configs,args)

    # set the dataloader list, model list, optimizer list, optimizer schedule list
    models = []
    classifiers = []
    optimizers = []
    classifier_optimizers = []
    optimizer_schedulers = []
    classifier_optimizer_schedulers = []

    model_s = FNet(configs["ModelConfig"]["backbone"], args.bn_momentum,
                                    pretrained=configs["ModelConfig"]["pretrained"],
                                    data_parallel=args.data_parallel).cuda()
    model_t = FNet(configs["ModelConfig"]["backbone"], args.bn_momentum,
                                    pretrained=configs["ModelConfig"]["pretrained"],
                                    data_parallel=args.data_parallel).cuda()
    classifier_s = FNetClassifier_s(configs["ModelConfig"]["backbone"], 345, args.data_parallel).cuda()
    classifier_t = FNetClassifier_t(configs["ModelConfig"]["backbone"], 345, args.data_parallel).cuda()
    models = [model_s, model_t]
    classifiers = [classifier_s, classifier_t]
    # federated learning step 1: initialize model with the same parameter (use target as standard)
    for source_weight, target_weight in zip(model_s.named_parameters(), model_t.named_parameters()):
        # consistent parameters
        source_weight[1].data = target_weight[1].data.clone()

    # create the optimizer for each model
    for model in models:
        optimizers.append(
            torch.optim.SGD(model.parameters(), momentum=args.momentum,
                            lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd))
    for classifier in classifiers:
        classifier_optimizers.append(
            torch.optim.SGD(classifier.parameters(), momentum=args.momentum,
                            lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd))
    # create the optimizer scheduler with cosine annealing schedule
    for optimizer in optimizers:
        optimizer_schedulers.append(
            CosineAnnealingLR(optimizer, configs["TrainingConfig"]["total_epochs"],
                              eta_min=configs["TrainingConfig"]["learning_rate_end"]))
    for classifier_optimizer in classifier_optimizers:
        classifier_optimizer_schedulers.append(
            CosineAnnealingLR(classifier_optimizer, configs["TrainingConfig"]["total_epochs"],
                              eta_min=configs["TrainingConfig"]["learning_rate_end"]))

    # create the event to save log info
    writer_log_dir = path.join(args.base_path, configs["DataConfig"]["dataset"], "runs",
                               "train_time:{}".format(args.train_time) + "_" +
                               args.target_domain + "_" + "_".join(args.source_domains))
    print("create writer in {}".format(writer_log_dir))
    if os.path.exists(writer_log_dir):
        flag = input("{} train_time:{} will be removed, input yes to continue:".format(
            configs["DataConfig"]["dataset"], args.train_time))
        if flag == "yes":
            shutil.rmtree(writer_log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=writer_log_dir)
    # begin train
    print("Begin the {} time's training, Dataset:{}, Source Domains {}, Target Domain {}".format(args.train_time,
                                                                                                 configs[
                                                                                                     "DataConfig"][
                                                                                                     "dataset"],
                                                                                                 args.source_domains,
                                                                                                 args.target_domain))
    total_epochs=configs["TrainingConfig"]["total_epochs"]
    batch_size=configs["TrainingConfig"]["batch_size"]

    # # train model
    #self supervised learning train

    #self supervised learning train
    train(train_dloaders, models, classifiers, optimizers,
                              classifier_optimizers, epoch, writer, num_classes=num_classes,
                              source_domains=args.source_domains,
                              batch_size=batch_size, total_epochs=total_epochs,
                              batchnorm_mmd=configs["UMDAConfig"]["batchnorm_mmd"],
                              pretrain_epoch = 200)

    #     test(args.target_domain, args.source_domains, test_dloaders, models, classifiers, epoch,
    #          writer, num_classes=num_classes, top_5_accuracy=(num_classes > 10))
    #     for scheduler in optimizer_schedulers:
    #         scheduler.step(epoch)
    #     for scheduler in classifier_optimizer_schedulers:
    #         scheduler.step(epoch)
    #     # save models every 10 epochs
    #     if (epoch + 1) % 10 == 0:
    #         # save target model with epoch, domain, model, optimizer
    #         save_checkpoint(
    #             {"epoch": epoch + 1,
    #              "domain": args.target_domain,
    #              "backbone": models[0].state_dict(),
    #              "classifier": classifiers[0].state_dict(),
    #              "optimizer": optimizers[0].state_dict(),
    #              "classifier_optimizer": classifier_optimizers[0].state_dict()
    #              },
    #             filename="{}.pth.tar".format(args.target_domain))


def save_checkpoint(state, filename):
    filefolder = "{}/{}/parameter/train_time:{}".format(args.base_path, configs["DataConfig"]["dataset"],
                                                        args.train_time)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))


if __name__ == "__main__":
    main()
