import sys
import os
import math
import time
from copy import copy
import tempfile
import builtins
import yaml
import logging
from easydict import EasyDict
from cvtorchvision import cvtransforms

# import cvtransforms

from ..trainer import Trainer
from ..exception import TrainerException
from ..utils import fix_random_seeds

from aiearth.deeplearning.datasets import Bigearthnet, LMDBDataset, random_subset
from aiearth.deeplearning.trainer.multispectral.models.vits import vit_small
from aiearth.deeplearning.trainer.multispectral.models.swin import swin_tiny
from aiearth.deeplearning.cloud.model import AIEModel

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

# _logger = logging.getLogger("mscls_train")


class AIEMSclsCloudModel(AIEModel):
    CLOUD_MODEL_TYPE = None

    def __init__(self):
        super.__init__()
        # _logger.info(
        #     "MultiSpectral Classfication is not supportted to train with CLOUD now."
        # )

    def get_mean(self):
        pass

    def get_std(self):
        pass

    def get_onnx_shape(self):
        pass

    def get_classes(self):
        pass

    def get_model_type(self):
        pass

    def is_fp16(self):
        pass


def init_distributed_mode(cfg):
    if cfg.distributed is None:
        cfg.distributed = True if "MASTER_ADDR" in os.environ else False

    if cfg.distributed:
        cfg.is_slurm_job = "SLURM_JOB_ID" in os.environ

        if cfg.is_slurm_job:
            cfg.rank = int(os.environ["SLURM_PROCID"])
            cfg.world_size = int(os.environ["SLURM_NNODES"]) * int(
                os.environ["SLURM_TASKS_PER_NODE"][0]
            )
        else:
            # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
            # read environment variables
            cfg.rank = int(os.environ["RANK"])
            cfg.world_size = int(os.environ["WORLD_SIZE"])

        # prepare distributed
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=cfg.dist_url,
            world_size=cfg.world_size,
            rank=cfg.rank,
        )

        # set cuda device
    cfg.gpu_to_work_on = cfg.rank % torch.cuda.device_count()
    torch.cuda.set_device(cfg.gpu_to_work_on)
    return cfg.distributed


def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate based on schedule"""
    lr = cfg.lr
    if cfg.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / cfg.epochs))
    else:  # stepwise lr schedule
        for milestone in cfg.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class MSClsTrainer(Trainer):
    config_base_dir = None

    def __init__(self, config_name, work_dir="./workspace") -> None:
        cfg = yaml.load(
            open(os.path.join(os.path.dirname(__file__), "configs", config_name)),
            Loader=yaml.Loader,
        )
        self.cfg = EasyDict(cfg)
        self.cfg.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

        init_distributed_mode(self.cfg)
        fix_random_seeds(self.cfg.seed)
        if self.cfg.rank == 0 and not os.path.isdir(self.cfg.checkpoints_dir):
            os.makedirs(self.cfg.checkpoints_dir)
        # if self.cfg.rank == 0:
        try:
            self.tb_writer = SummaryWriter(
                os.path.join(self.cfg.work_dir, "log")
            )
        except:
            pass
        # if self.cfg.rank == 0:
        logger=logging.getLogger("Traing AID by moco pretrained model")
        logger.setLevel(logging.DEBUG)
        ls=logging.StreamHandler()
        ls.setLevel(logging.DEBUG)
        formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
        ls.setFormatter(formatter)
        logger.addHandler(ls)
        logfile=os.path.join(self.cfg.checkpoints_dir,"log.txt")
        lf=logging.FileHandler(filename=logfile,encoding="utf8")
        lf.setLevel(logging.DEBUG)
        lf.setFormatter(formatter)
        logger.addHandler(lf)   
        self._logger=logger

    def to_cloud_model(self, onnx_shape=None) -> AIEMSclsCloudModel:
        return AIEMSclsCloudModel(self.cfg, onnx_shape)

    def export_onnx(self, output_file=None, checkpoint_path=None, shape=(1024, 1024)):
        if not checkpoint_path:
            checkpoint_path = os.path.join(self.work_dir, "latest.pth")
        if not output_file:
            output_file = checkpoint_path.replace(".pth", ".onnx")
        # to_onnx(self.cfg, output_file, checkpoint_path, shape)
        return output_file

    def build_loader(self):
        if self.cfg.bands == "RGB":
            bands = ["B04", "B03", "B02"]
            lmdb_train = "train_RGB.lmdb"
            lmdb_val = "val_RGB.lmdb"
        else:
            bands = [
                "B01",
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B09",
                "B11",
                "B12",
            ]
            lmdb_train = "train_B12.lmdb"
            lmdb_val = "val_B12.lmdb"

        train_transforms = cvtransforms.Compose(
            [
                cvtransforms.RandomResizedCrop(
                    224, scale=(0.8, 1.0)
                ),  # multilabel, avoid cropping out labels
                cvtransforms.RandomHorizontalFlip(),
                cvtransforms.ToTensor(),
            ]
        )

        val_transforms = cvtransforms.Compose(
            [
                cvtransforms.Resize(256),
                cvtransforms.CenterCrop(224),
                cvtransforms.ToTensor(),
            ]
        )

        if self.cfg.lmdb:
            train_dataset = LMDBDataset(
                lmdb_file=os.path.join(self.cfg.lmdb_dir, lmdb_train),
                transform=train_transforms,
            )

            val_dataset = LMDBDataset(
                lmdb_file=os.path.join(self.cfg.lmdb_dir, lmdb_val),
                transform=val_transforms,
            )
        else:
            train_dataset = Bigearthnet(
                root=self.cfg.data_dir,
                split="train",
                bands=bands,
                use_new_labels=True,
                transform=train_transforms,
            )

            val_dataset = Bigearthnet(
                root=self.cfg.data_dir,
                split="val",
                bands=bands,
                use_new_labels=True,
                transform=train_transforms,
            )

        if self.cfg.train_frac is not None and self.cfg.train_frac < 1:
            train_dataset = random_subset(
                train_dataset, self.cfg.train_frac, self.cfg.seed
            )

        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            # shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.is_slurm_job,  # improve a little when using lmdb dataset
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.is_slurm_job,  # improve a little when using lmdb dataset
            drop_last=True,
        )

        self._logger.info(
            "train_len: %d val_len: %d" % (len(train_dataset), len(val_dataset))
        )

        return train_dataset, val_dataset, train_loader, val_loader

    def set_classes(self, classes_list):
        self.cfg["aie_classes"] = classes_list
        return

    def load_from(self, pretrained_model):
        if type(pretrained_model) == str:
            self.cfg.load_from = pretrained_model
        elif hasattr(pretrained_model, "local_path"):
            self.cfg.load_from = pretrained_model.local_path

    def build_model(self,test_checkpoint=None):
        if self.cfg.backbone == "resnet50":
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(2048, 19)
            linear_keyword = 'fc'
        elif self.cfg.backbone == "resnet18":
            model = models.resnet18(pretrained=False)
            model.fc = torch.nn.Linear(512, 19)
            linear_keyword = 'fc'
        elif self.cfg.backbone == "vit_small":
            model = vit_small(in_chans=13,num_classes=19)
            linear_keyword = 'head'
        elif self.cfg.backbone == "swin_tiny":
            model=swin_tiny(in_chans=13,num_classes=19)
            linear_keyword = 'head'
        else:
            print("we only support res18, res50, vit-small, and swin-tiny as our model now!!")
            exit()

        if self.cfg.linear:
            # freeze all layers but the last fc
            for name, param in model.named_parameters():
                if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
                    param.requires_grad = False
            # init the fc layer
            getattr(model, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
            getattr(model, linear_keyword).bias.data.zero_()       
            

        if self.cfg.bands == "all":
            model.conv1 = torch.nn.Conv2d(
                13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

#         if self.cfg.linear:
#             for name, param in model.named_parameters():
#                 if name not in ["fc.weight", "fc.bias"]:
#                     param.requires_grad = False

#             model.fc.weight.data.normal_(mean=0.0, std=0.01)
#             model.fc.bias.data.zero_()

        # load from pre-trained, before DistributedDataParallel constructor
        if self.cfg.pretrained:
            if os.path.isfile(self.cfg.pretrained):
                self._logger.info("=> loading checkpoint '{}'".format(self.cfg.pretrained))
                checkpoint = torch.load(self.cfg.pretrained, map_location="cpu")

                # rename moco pre-trained keys
                if "swin" in self.cfg.backbone:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint["state_dict"]
                # _logger.info(state_dict.keys())
                for k in list(state_dict.keys()):
                    
                    if "resnet" in self.cfg.backbone:
                        # retain only encoder up to before the embedding layer
                        if k.startswith("module.encoder_q") and not k.startswith(
                            "module.encoder_q.fc"
                        ):
                            # pdb.set_trace()
                            # remove prefix
                            state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
                        # delete renamed or unused k
                    elif "vit" in self.cfg.backbone:
                        # retain only base_encoder up to before the embedding layer
                        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                            # remove prefix
                            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                        # delete renamed or unused k
                    elif "swin" in self.cfg.backbone:
                        if k.startswith('encoder.'):
                            # remove prefix
                            state_dict[k[len("encoder."):]] = state_dict[k]
                    else:
                        print("not supporting model")
                        exit()

                    del state_dict[k]

                """
                # remove prefix
                state_dict = {k.replace("module.", ""): v for k,v in state_dict.items()}
                """
                # cfg.start_epoch = 0
                msg = model.load_state_dict(state_dict, strict=False)
                # pdb.set_trace()
                # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

                self._logger.info(
                    "=> loaded pre-trained model '{}'".format(self.cfg.pretrained)
                )
            else:
                self._logger.info(
                    "=> no checkpoint found at '{}'".format(self.cfg.pretrained)
                )
        if test_checkpoint:
            print("testing process, loading weight from '{}'".format(test_checkpoint))
            checkpoint = torch.load(test_checkpoint)
            state_dict = checkpoint["model_state_dict"]
            # state_dict = {k.replace("module.", ""): v for k,v in state_dict.items()}
            msg=model.load_state_dict(state_dict)
            # print(msg)
            # exit()

        # convert batch norm layers (if any)
        if self.cfg.is_slurm_job:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model.cuda()
        if self.cfg.is_slurm_job:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.cfg.gpu_to_work_on], find_unused_parameters=True
            )

        criterion = torch.nn.MultiLabelSoftMarginLoss()
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        if "resnet" in self.cfg.backbone:
            optimizer = torch.optim.SGD(parameters, lr=self.cfg.lr, momentum=0.9)
        else:
            optimizer = torch.optim.SGD(parameters, self.cfg.lr,
                                        momentum=self.cfg.momentum,
                                        weight_decay=self.cfg.weight_decay)            

        start_epoch = 0
        if self.cfg.resume:
            checkpoint = torch.load(self.cfg.resume)
            state_dict = checkpoint["model_state_dict"]
            # state_dict = {k.replace("module.", ""): v for k,v in state_dict.items()}
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]

        return model, criterion, optimizer, start_epoch

    @classmethod
    def list_config(cls):
        files = os.listdir(cls.config_base_dir)
        return [f.strip(".yaml") for f in files if f.endswith(".yaml")]

    def get_onnx_script(self):
        package_base_dir = os.path.dirname(sys.modules["train"].__file__)
        return os.path.join(package_base_dir, self.onnx_script)

    def train(self, validate=False, distributed=None):
        # Build the dataset
        train_dataset, val_dataset, train_loader, val_loader = self.build_loader()
        # Build the detector
        model, criterion, optimizer, start_epoch = self.build_model()
        self._logger.info("Start training...")
        for epoch in range(start_epoch, self.cfg.epochs):
            model.train()
            adjust_learning_rate(optimizer, epoch, self.cfg)

            train_loader.sampler.set_epoch(epoch)
            running_loss = 0.0
            running_acc = 0.0

            running_loss_epoch = 0.0
            running_acc_epoch = 0.0

            start_time = time.time()
            end = time.time()
            sum_bt = 0.0
            sum_dt = 0.0
            sum_tt = 0.0
            sum_st = 0.0
            for i, data in enumerate(train_loader, 0):
                data_time = time.time() - end
                # inputs, labels = data
                b_zeros = torch.zeros(
                    (data[0].shape[0], 1, data[0].shape[2], data[0].shape[3]),
                    dtype=torch.float32,
                )
                images = torch.cat(
                    (data[0][:, :10, :, :], b_zeros, data[0][:, 10:, :, :]), dim=1
                )
                # inputs, labels = data[0].cuda(), data[1].cuda()
                inputs, labels = images.cuda(), data[1].cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                # pdb.set_trace()
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()
                train_time = time.time() - end - data_time
                if epoch % 5 == 4:
                    score = torch.sigmoid(outputs).detach().cpu()
                    average_precision = (
                        average_precision_score(labels.cpu(), score, average="micro")
                        * 100.0
                    )
                else:
                    average_precision = 0
                score_time = time.time() - end - data_time - train_time

                # print statistics
                running_loss += loss.item()
                running_acc += average_precision
                batch_time = time.time() - end
                end = time.time()
                sum_bt += batch_time
                sum_dt += data_time
                sum_tt += train_time
                sum_st += score_time

                if i % self.cfg.print_freq == 0:  # print every 20 mini-batches
                    self._logger.info(
                        "[%d, %5d] loss: %.3f acc: %.3f batch_time: %.3f data_time: %.3f train_time: %.3f score_time: %.3f"
                        % (
                            epoch + 1,
                            i + 1,
                            running_loss / self.cfg.print_freq,
                            running_acc / self.cfg.print_freq,
                            sum_bt / self.cfg.print_freq,
                            sum_dt / self.cfg.print_freq,
                            sum_tt / self.cfg.print_freq,
                            sum_st / self.cfg.print_freq,
                        )
                    )

                    # train_iter =  i*args.batch_size / len(train_dataset)
                    # tb_writer.add_scalar('train_loss', running_loss/20, global_step=(epoch+1+train_iter) )
                    running_loss_epoch = running_loss / self.cfg.print_freq
                    running_acc_epoch = running_acc / self.cfg.print_freq

                    running_loss = 0.0
                    running_acc = 0.0
                    sum_bt = 0.0
                    sum_dt = 0.0
                    sum_tt = 0.0
                    sum_st = 0.0

            running_loss_val = 0.0
            running_acc_val = 0.0
            count_val = 0
            model.eval()
            with torch.no_grad():
                for j, data_val in enumerate(val_loader, 0):
                    b_zeros = torch.zeros(
                        (
                            data_val[0].shape[0],
                            1,
                            data_val[0].shape[2],
                            data_val[0].shape[3],
                        ),
                        dtype=torch.float32,
                    )
                    images = torch.cat(
                        (
                            data_val[0][:, :10, :, :],
                            b_zeros,
                            data_val[0][:, 10:, :, :],
                        ),
                        dim=1,
                    )

                    # inputs_val, labels_val = data_val[0].cuda(), data_val[1].cuda()
                    inputs_val, labels_val = images.cuda(), data_val[1].cuda()

                    outputs_val = model(inputs_val)
                    loss_val = criterion(outputs_val, labels_val.long())
                    score_val = torch.sigmoid(outputs_val).detach().cpu()
                    average_precision_val = (
                        average_precision_score(
                            labels_val.cpu(), score_val, average="micro"
                        )
                        * 100.0
                    )

                    count_val += 1
                    running_loss_val += loss_val.item()
                    running_acc_val += average_precision_val

            self._logger.info(
                "Epoch %d val_loss: %.3f val_acc: %.3f time: %s seconds."
                % (
                    epoch + 1,
                    running_loss_val / count_val,
                    running_acc_val / count_val,
                    time.time() - start_time,
                )
            )

            if self.cfg.rank == 0:
                losses = {
                    "train": running_loss_epoch,
                    "val": running_loss_val / count_val,
                }
                accs = {
                    "train": running_acc_epoch,
                    "val": running_acc_val / count_val,
                }
                self.tb_writer.add_scalars(
                    "loss", losses, global_step=epoch + 1, walltime=None
                )
                self.tb_writer.add_scalars(
                    "acc", accs, global_step=epoch + 1, walltime=None
                )
                # print(epoch + 1,self.cfg.save_frep)
                if (epoch + 1) % self.cfg.save_frep == 0:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss,
                        },
                        os.path.join(
                            self.cfg.checkpoints_dir,
                            "checkpoint_{:04d}.pth.tar".format(epoch),
                        ),
                    )
                    # exit()
            # exit()
        # if args.rank==0:
        #    torch.save(net.state_dict(), save_path)

        self._logger.info("Training finished.")

    # def test(self, checkpoint, output_dir=None, eval=False, metrics=[]):
    def test(self,checkpoint):
        # Build the dataset
        _, val_dataset, _, val_loader = self.build_loader()
        # Build the detector
        start_time = time.time()
        model,criterion, _, _ = self.build_model(checkpoint)
        self._logger.info("Start testing...")

        running_loss_val = 0.0
        running_acc_val = 0.0
        count_val = 0
        model.eval()
        with torch.no_grad():
            for j, data_val in enumerate(val_loader, 0):
                b_zeros = torch.zeros(
                    (
                        data_val[0].shape[0],
                        1,
                        data_val[0].shape[2],
                        data_val[0].shape[3],
                    ),
                    dtype=torch.float32,
                )
                images = torch.cat(
                    (
                        data_val[0][:, :10, :, :],
                        b_zeros,
                        data_val[0][:, 10:, :, :],
                    ),
                    dim=1,
                )

                # inputs_val, labels_val = data_val[0].cuda(), data_val[1].cuda()
                inputs_val, labels_val = images.cuda(), data_val[1].cuda()

                outputs_val = model(inputs_val)
                loss_val = criterion(outputs_val, labels_val.long())
                score_val = torch.sigmoid(outputs_val).detach().cpu()
                average_precision_val = (
                    average_precision_score(
                        labels_val.cpu(), score_val, average="micro"
                    )
                    * 100.0
                )

                count_val += 1
                running_loss_val += loss_val.item()
                running_acc_val += average_precision_val

            self._logger.info(
                "val_loss: %.3f val_acc: %.3f time: %s seconds."
                % (
                    running_loss_val / count_val,
                    running_acc_val / count_val,
                    time.time() - start_time,
                )
            )

        self._logger.info("Testing finished.")
