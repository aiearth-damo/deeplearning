# Copyright (c) OpenMMLab. All rights reserved.
# -*- conding: utf-8 -*-
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry("models")

# from mmseg.models.builder import MODELS

ALGORITHMS = MODELS
MUTABLES = MODELS
DISTILLERS = MODELS
DISTILLERSLOSSES = MODELS
OPS = MODELS
PRUNERS = MODELS
QUANTIZERS = MODELS
ARCHITECTURES = MODELS
MUTATORS = MODELS


def build_algorithm(cfg):
    """Build compressor."""
    return ALGORITHMS.build(cfg)


def build_architecture(cfg):
    """Build architecture."""
    return ARCHITECTURES.build(cfg)


def build_mutator(cfg):
    """Build mutator."""
    return MUTATORS.build(cfg)


def build_distiller(cfg):
    """Build distiller."""
    return DISTILLERS.build(cfg)


def build_pruner(cfg):
    """Build pruner."""
    return PRUNERS.build(cfg)


def build_mutable(cfg):
    """Build mutable."""
    return MUTABLES.build(cfg)


def build_op(cfg):
    """Build op."""
    return OPS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return DISTILLERSLOSSES.build(cfg)
