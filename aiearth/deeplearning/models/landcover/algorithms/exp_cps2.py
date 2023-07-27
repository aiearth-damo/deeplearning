# Copyright (c) OpenMMLab. All rights reserved.
# -*- conding: utf-8 -*-
from torch import nn
import copy
import torch

from mmseg.ops import resize
from mmseg.core import add_prefix

from ..builder import ALGORITHMS, build_architecture, build_loss
from .semi_cps import GeneralSemiBase
from .semantic_estimator import semantic_estimator


@ALGORITHMS.register_module()
class ExpGeneralSemiCPSFDOnlineEMAOnce3CutmixSWFD(GeneralSemiBase):
    def __init__(
        self,
        components=dict(),
        resume=None,
        end_momentum=None,
        # start_semi_iter=0,
        update="none",
        **kwargs,
    ):
        super(ExpGeneralSemiCPSFDOnlineEMAOnce3CutmixSWFD, self).__init__(**kwargs)
        self.components = components
        self.outputs = dict()
        self.losses = nn.ModuleDict()
        for i, component in enumerate(self.components):
            self.outputs[component["module"]] = list()
            self.outputs[component["module"] + ".features"] = list()
            for loss in component.losses:
                loss_cfg = loss.copy()
                loss_name = loss_cfg.pop("name")
                assert "loss" in loss_name, "loss must in name, but get {}".format(
                    loss_name
                )
                self.losses[loss_name] = build_loss(loss_cfg)

        self.architecture = build_architecture(copy.deepcopy(self.architecture_cfg))
        self.prepare_from_model(self.architecture, self.components, self.outputs)
        self.architecture2 = build_architecture(copy.deepcopy(self.architecture_cfg))
        self.prepare_from_model(self.architecture2, self.components, self.outputs)

        self.teacher_architecture = build_architecture(
            copy.deepcopy(self.architecture_cfg)
        )
        self.prepare_from_model(
            self.teacher_architecture, self.components, self.outputs
        )
        self.teacher_architecture2 = build_architecture(
            copy.deepcopy(self.architecture_cfg)
        )
        self.prepare_from_model(
            self.teacher_architecture2, self.components, self.outputs
        )

        self.estimator_branch1 = semantic_estimator(
            self.architecture.model.decode_head.channels,
            self.architecture.model.decode_head.num_classes,
            ignore_label=self.architecture.model.decode_head.ignore_index,
            resume=resume,
        )
        self.estimator_branch2 = semantic_estimator(
            self.architecture.model.decode_head.channels,
            self.architecture.model.decode_head.num_classes,
            ignore_label=self.architecture.model.decode_head.ignore_index,
            resume=resume,
        )
        self.estimator_teacher1 = semantic_estimator(
            self.architecture.model.decode_head.channels,
            self.architecture.model.decode_head.num_classes,
            ignore_label=self.architecture.model.decode_head.ignore_index,
            resume=resume,
        )
        self.estimator_teacher2 = semantic_estimator(
            self.architecture.model.decode_head.channels,
            self.architecture.model.decode_head.num_classes,
            ignore_label=self.architecture.model.decode_head.ignore_index,
            resume=resume,
        )
        self.current_iter = 0
        # self.start_semi_iter = start_semi_iter
        self.momentum_init(self.architecture, self.teacher_architecture)
        self.momentum_init(self.architecture2, self.teacher_architecture2)
        self.end_momentum = end_momentum
        self.update = update

    @torch.no_grad()
    def momentum_init(self, online_net, target_net):
        """target_net is teacher network"""
        for param_ol, param_tgt in zip(
            online_net.parameters(), target_net.parameters()
        ):
            param_tgt.data.copy_(param_ol.data)
            param_tgt.requires_grad = False

    @torch.no_grad()
    def momentum_update(self, online_net, target_net, momentum):
        """Momentum update of the target network."""
        for param_ol, param_tgt in zip(
            online_net.parameters(), target_net.parameters()
        ):
            param_tgt.data = param_tgt.data * momentum + param_ol.data * (
                1.0 - momentum
            )

    def collect(self, model, data, output, features, keys):
        loss = self.exec_forward(model, data, output)
        features.update(add_prefix(output, keys))
        return loss

    def train_step(self, data, optimizer):
        """"""
        self.current_iter = self.current_iter + 1
        if self.update == "none":
            self.momentum = min(1 - 1 / self.current_iter, self.end_momentum)
        elif self.update == "liner":
            self.momentum = (1 - 1 / self.current_iter) * self.end_momentum
        else:
            NotImplementedError("error")
        self.momentum_update(
            self.architecture, self.teacher_architecture, self.momentum
        )
        self.momentum_update(
            self.architecture2, self.teacher_architecture2, self.momentum
        )
        losses = dict()
        outputs_features = dict()
        if "cutmask" in data["sup_data"].keys():
            sup_cutmix = data["sup_data"].pop("cutmask")
            sup_cutmix = sup_cutmix.unsqueeze(1).float()

        unsup_cutmix = data["unsup_data"]["weak"].pop("cutmask")
        # unsup_cutmix2 = data["unsup_data"]["strong"].pop("cutmask")
        unsup_cutmix = unsup_cutmix.unsqueeze(1).float()
        unsup_imgs = data["unsup_data"]["strong"]["img"]
        assert unsup_imgs.size()[0] >= 2
        unsup_rand_index = list(range(unsup_imgs.size()[0]))[1:] + [0]
        outputs_features["unsup_cutmix"] = unsup_cutmix
        outputs_features["unsup_rand_index"] = unsup_rand_index
        data["unsup_data"]["weak"]["gt_semantic_seg"] = torch.zeros_like(
            data["unsup_data"]["weak"]["gt_semantic_seg"]
        )
        data["unsup_data"]["strong"]["gt_semantic_seg"] = torch.zeros_like(
            data["unsup_data"]["strong"]["gt_semantic_seg"]
        )

        unsup_data_mix = {
            "img_metas": data["unsup_data"]["strong"]["img_metas"],
            "img": unsup_cutmix * unsup_imgs
            + (1 - unsup_cutmix) * unsup_imgs[unsup_rand_index],
            "gt_semantic_seg": data["unsup_data"]["strong"]["gt_semantic_seg"],
        }

        with torch.no_grad():
            self.collect(
                self.teacher_architecture,
                data["sup_data"],
                self.outputs,
                outputs_features,
                "teacher1.sup",
            )
            self.collect(
                self.teacher_architecture,
                data["unsup_data"]["weak"],
                self.outputs,
                outputs_features,
                "teacher1.unsup",
            )
            self.collect(
                self.teacher_architecture2,
                data["sup_data"],
                self.outputs,
                outputs_features,
                "teacher2.sup",
            )
            self.collect(
                self.teacher_architecture2,
                data["unsup_data"]["weak"],
                self.outputs,
                outputs_features,
                "teacher2.unsup",
            )

        sup_losses1 = self.collect(
            self.architecture,
            data["sup_data"],
            self.outputs,
            outputs_features,
            "branch1.sup",
        )
        _ = self.collect(
            self.architecture,
            unsup_data_mix,
            self.outputs,
            outputs_features,
            "branch1.unsup",
        )

        sup_losses2 = self.collect(
            self.architecture2,
            data["sup_data"],
            self.outputs,
            outputs_features,
            "branch2.sup",
        )
        _ = self.collect(
            self.architecture2,
            unsup_data_mix,
            self.outputs,
            outputs_features,
            "branch2.unsup",
        )

        losses.update(add_prefix(sup_losses1, "sup1"))
        losses.update(add_prefix(sup_losses2, "sup2"))

        unsup_losses = self.compute_semi_loss(
            data,
            outputs_features,
            self.architecture.model.decode_head.conv_seg,
            self.architecture2.model.decode_head.conv_seg,
            self.teacher_architecture.model.decode_head.conv_seg,
            self.teacher_architecture2.model.decode_head.conv_seg,
            self.estimator_branch1,
            self.estimator_branch2,
            self.estimator_teacher1,
            self.estimator_teacher2,
            self.current_iter,
            data["sup_data"]["gt_semantic_seg"],
        )

        losses.update(add_prefix(unsup_losses, "unsup"))

        losses["momentum"] = self.momentum * torch.ones(1).cuda()
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data["sup_data"]["img"].data)
        )

        return outputs

    def compute_semi_loss(
        self,
        data,
        features,
        conv_seg1,
        conv_seg2,
        teacher_conv_seg1,
        teacher_conv_seg2,
        estimator_branch1,
        estimator_branch2,
        estimator_teacher1,
        estimator_teacher2,
        current_iter,
        gt_semantic_seg,
    ):
        losses = dict()
        for i, component in enumerate(self.components):
            module_name = component["module"]
            unsup_cutmix = features["unsup_cutmix"]
            unsup_rand_index = features["unsup_rand_index"]
            shape = features[f"teacher1.unsup.{module_name}"][0].shape
            unsup_cutmix = resize(unsup_cutmix, size=shape[2:], mode="nearest")
            teacher1_unsup = [
                unsup_cutmix * x + (1 - unsup_cutmix) * x[unsup_rand_index]
                for x in features[f"teacher1.unsup.{module_name}"]
            ]
            teacher2_unsup = [
                unsup_cutmix * x + (1 - unsup_cutmix) * x[unsup_rand_index]
                for x in features[f"teacher2.unsup.{module_name}"]
            ]
            teacher1_unsup_features = [
                unsup_cutmix * x + (1 - unsup_cutmix) * x[unsup_rand_index]
                for x in features[f"teacher1.unsup.{module_name}.features"]
            ]
            teacher2_unsup_features = [
                unsup_cutmix * x + (1 - unsup_cutmix) * x[unsup_rand_index]
                for x in features[f"teacher2.unsup.{module_name}.features"]
            ]

            teacher1_logits = [
                torch.cat(
                    features[f"teacher1.sup.{module_name}"] + teacher1_unsup, dim=0
                )
            ]
            teacher2_logits = [
                torch.cat(
                    features[f"teacher2.sup.{module_name}"] + teacher2_unsup, dim=0
                )
            ]
            branch1_logtits = [
                torch.cat(
                    features[f"branch1.sup.{module_name}"]
                    + features[f"branch1.unsup.{module_name}"],
                    dim=0,
                )
            ]
            branch2_logtits = [
                torch.cat(
                    features[f"branch2.sup.{module_name}"]
                    + features[f"branch2.unsup.{module_name}"],
                    dim=0,
                )
            ]

            teacher1_features = [
                torch.cat(
                    features[f"teacher1.sup.{module_name}.features"]
                    + teacher1_unsup_features,
                    dim=0,
                )
            ]
            teacher2_features = [
                torch.cat(
                    features[f"teacher2.sup.{module_name}.features"]
                    + teacher2_unsup_features,
                    dim=0,
                )
            ]
            branch1_features = [
                torch.cat(
                    features[f"branch1.sup.{module_name}.features"]
                    + features[f"branch1.unsup.{module_name}.features"],
                    dim=0,
                )
            ]
            branch2_features = [
                torch.cat(
                    features[f"branch2.sup.{module_name}.features"]
                    + features[f"branch2.unsup.{module_name}.features"],
                    dim=0,
                )
            ]

            for out_idx, (
                o_branch1_logits,
                o_branch2_logtis,
                o_teacher1_logits,
                o_teacher2_logits,
                o_branch1_features,
                o_branch2_features,
                o_teacher1_features,
                o_teacher2_features,
            ) in enumerate(
                zip(
                    branch1_logtits,
                    branch2_logtits,
                    teacher1_logits,
                    teacher2_logits,
                    branch1_features,
                    branch2_features,
                    teacher1_features,
                    teacher2_features,
                )
            ):
                for loss in component.losses:
                    loss_module = self.losses[loss.name]
                    loss_name = f"{loss.name}.{out_idx}"

                    loss_module.current_data = data
                    loss, ratio = loss_module(
                        o_branch1_logits,
                        o_branch1_features,
                        conv_seg1,
                        estimator_branch1,
                        o_branch2_logtis,
                        o_branch2_features,
                        conv_seg2,
                        estimator_branch2,
                        o_teacher1_logits,
                        o_teacher1_features,
                        teacher_conv_seg1,
                        estimator_teacher1,
                        o_teacher2_logits,
                        o_teacher2_features,
                        teacher_conv_seg2,
                        estimator_teacher2,
                        current_iter,
                        gt_semantic_seg,
                    )
                    losses[loss_name] = loss
                    losses[loss_name.replace("loss", "ratio")] = ratio
                    loss_module.current_data = None

        return losses

    def prepare_from_model(self, model, components, output_features):
        module2name = dict()
        for name, module in model.model.named_modules():
            module2name[module] = name
        name2module = dict(model.model.named_modules())

        def forward_output_hook(module, inputs, outputs):
            if self.training:
                output_features[module2name[module]].append(outputs)
                output_features[module2name[module] + ".features"].append(inputs[0])

        for component in components:
            name2module[component["module"]].register_forward_hook(forward_output_hook)
