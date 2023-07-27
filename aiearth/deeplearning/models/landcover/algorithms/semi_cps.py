# Copyright (c) OpenMMLab. All rights reserved.
# -*- conding: utf-8 -*-
import copy
import torch
from torch import nn
import gc

from ..builder import ALGORITHMS, build_architecture, build_loss
from .base import BaseAlgorithm
from mmseg.core import add_prefix


@ALGORITHMS.register_module()
class GeneralSemiBase(BaseAlgorithm):
    def train_step(self, data, optimizer):
        pass

    def prepare_from_model(self, model, components, output_features):
        module2name = dict()
        for name, module in model.model.named_modules():
            module2name[module] = name
        name2module = dict(model.model.named_modules())

        def forward_output_hook(module, inputs, outputs):
            if self.training:
                output_features[module2name[module]].append(outputs)
                # output_features["inputs_"+module2name[module]].append(inputs)

        for component in components:
            name2module[component["module"]].register_forward_hook(forward_output_hook)

    def reset_outputs(self, outputs):
        for key in outputs.keys():
            outputs[key] = list()

    def exec_forward(self, model, data, output_features):
        # Clear the saved data of the last forward。
        self.reset_outputs(output_features)
        output = model(**data)
        return output


@ALGORITHMS.register_module()
class GeneralSemiCPS(GeneralSemiBase):
    def __init__(self, components=dict(), **kwargs):
        super(GeneralSemiCPS, self).__init__(**kwargs)
        self.components = components
        self.outputs = dict()
        self.losses = nn.ModuleDict()
        for i, component in enumerate(self.components):
            self.outputs[component["module"]] = list()
            for loss in component.losses:
                loss_cfg = loss.copy()
                loss_name = loss_cfg.pop("name")
                assert "loss" in loss_name, "loss must in name, but get {}".format(
                    loss_name
                )
                self.losses[loss_name] = build_loss(loss_cfg)
        self.outputs2 = copy.deepcopy(self.outputs)
        self.prepare_from_model(self.architecture, self.components, self.outputs)
        self.architecture2 = build_architecture(copy.deepcopy(self.architecture_cfg))
        self.prepare_from_model(self.architecture2, self.components, self.outputs2)

    def train_step(self, data, optimizer):
        """"""
        losses = dict()
        outputs = dict()

        sup_losses2 = self.exec_forward(
            self.architecture2, data["sup_data"], self.outputs2
        )
        outputs.update(add_prefix(self.outputs2, "sup2"))
        sup_losses1 = self.exec_forward(
            self.architecture, data["sup_data"], self.outputs
        )
        outputs.update(add_prefix(self.outputs, "sup1"))

        tmp1 = self.exec_forward(self.architecture2, data["unsup_data"], self.outputs2)
        outputs.update(add_prefix(self.outputs2, "unsup2"))
        tmp2 = self.exec_forward(self.architecture, data["unsup_data"], self.outputs)
        outputs.update(add_prefix(self.outputs, "unsup1"))

        unsup_losses = self.compute_semi_loss(data, outputs)
        losses.update(add_prefix(unsup_losses, "unsup"))

        losses.update(add_prefix(sup_losses2, "sup1"))
        losses.update(add_prefix(sup_losses1, "sup2"))

        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data["sup_data"]["img"].data)
        )

        gc.collect()
        return outputs

    def compute_semi_loss(self, data, features):
        losses = dict()
        for i, component in enumerate(self.components):
            module_name = component["module"]
            unsup2 = features[f"unsup2.{module_name}"]
            unsup1 = features[f"unsup1.{module_name}"]
            sup2 = features[f"sup2.{module_name}"]
            sup1 = features[f"sup1.{module_name}"]
            branch1 = [torch.cat(unsup1 + sup1, dim=0)]
            branch2 = [torch.cat(unsup2 + sup2, dim=0)]

            for out_idx, (s_out, w_out) in enumerate(zip(branch1, branch2)):
                for loss in component.losses:
                    loss_module = self.losses[loss.name]
                    loss_name = f"{loss.name}.{out_idx}"

                    loss_module.current_data = data
                    losses[loss_name] = loss_module(s_out, w_out)
                    loss_module.current_data = None

        return losses
