# -*- conding: utf-8 -*-
import torch
from torch import nn

from mmcv.cnn.bricks.registry import NORM_LAYERS, CONV_LAYERS


@NORM_LAYERS.register_module(name="DSBatchNorm")
class DSBatchNorm(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        statistics_batch=None,
    ):
        """
        Deep Supervision Batch Normalization.
        """
        super().__init__()
        self.dsbn1 = nn.BatchNorm2d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.dsbn2 = nn.BatchNorm2d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.statistics_batch = statistics_batch
        # assert isinstance(self.statistics_batch, int)

    def forward(self, input):
        """
        Forward pass of the Deep Supervision Batch Normalization.
        """
        if self.training:
            if self.statistics_batch is None:
                B, C, H, W = input.shape
                assert B % 2 == 0, "error batch size {}".format(B)
                self.statistics_batch = B // 2
            input0 = self.dsbn1(input[: self.statistics_batch])
            input1 = self.dsbn2(input[self.statistics_batch:])
            output = torch.cat((input0, input1), 0)
        else:
            output = self.dsbn1(input)
        return output


@NORM_LAYERS.register_module(name="SyncDSBatchNorm")
class SyncDSBatchNorm(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        statistics_batch=None,
    ):
        """
        Synchronized Deep Supervision Batch Normalization.
        """
        super().__init__()
        self.dsbn1 = nn.SyncBatchNorm(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.dsbn2 = nn.SyncBatchNorm(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.statistics_batch = statistics_batch
        # assert isinstance(self.statistics_batch, int)

    def forward(self, input):
        """
        Forward pass of the Synchronized Deep Supervision Batch Normalization.
        """
        if self.training:
            if self.statistics_batch is None:
                B, C, H, W = input.shape
                assert B % 2 == 0, "error batch size {}".format(B)
                self.statistics_batch = B // 2
            input0 = self.dsbn1(input[: self.statistics_batch])
            input1 = self.dsbn2(input[self.statistics_batch:])
            output = torch.cat((input0, input1), 0)
        else:
            output = self.dsbn1(input)
        return output
