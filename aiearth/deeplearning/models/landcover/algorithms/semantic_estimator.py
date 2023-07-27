# -*- conding: utf-8 -*-
import torch
import torch.utils.data
import torch.distributed
import torch.backends.cudnn


class semantic_estimator(torch.nn.Module):
    def __init__(self, feature_num, num_classes, ignore_label=255, resume=None):
        super(semantic_estimator, self).__init__()

        self.class_num = num_classes
        self.feature_num = feature_num
        self.ignore_label = ignore_label
        # init mean and covariance
        self.register_buffer("CoVariance", torch.zeros(
            self.class_num, feature_num))
        self.register_buffer("Mean", torch.zeros(self.class_num, feature_num))
        self.register_buffer("Amount", torch.zeros(self.class_num))
        self.init(resume=resume)

    def update(self, features, labels):

        mask = labels != self.ignore_label
        # remove IGNORE_LABEL pixels
        labels = labels[mask]
        features = features[mask]

        N, A = features.size()
        C = self.class_num

        NxCxA_Features = features.view(N, 1, A).expand(N, C, A)

        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxA_Features.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        mean_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
            mean_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1).expand(C, A)
        )

        weight_CV[weight_CV != weight_CV] = 0

        additional_CV = weight_CV.mul(
            1 - weight_CV).mul((self.Mean - mean_CxA).pow(2))

        self.CoVariance = (
            self.CoVariance.mul(1 - weight_CV) + var_temp.mul(weight_CV)
        ).detach() + additional_CV.detach()

        self.Mean = (self.Mean.mul(1 - weight_CV) +
                     mean_CxA.mul(weight_CV)).detach()

        self.Amount = self.Amount + onehot.sum(0)

    def init(self, resume=None):
        if resume is not None:
            print("Loading checkpoint from {}".format(resume))
            checkpoint = torch.load(resume, map_location=torch.device("cpu"))
            self.CoVariance = checkpoint["state_dict"][
                "student_estimator.CoVariance"
            ].cuda(non_blocking=True)
            self.Mean = checkpoint["state_dict"]["student_estimator.Mean"].cuda(
                non_blocking=True
            )
            self.Amount = checkpoint["state_dict"]["student_estimator.Amount"].cuda(
                non_blocking=True
            )
            print(self.CoVariance)
            print(self.Mean)
        else:
            pass
