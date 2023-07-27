import argparse
from functools import partial
import os
from copy import copy

import cv2
import mmcv
import numpy as np
import onnxruntime as rt
import torch
import torch._C
import torch.serialization
from mmcv import DictAction
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint
from torch import nn
from onnx import helper

from mmseg.apis import show_result_pyplot
from mmseg.apis.inference import LoadImage
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor

torch.manual_seed(3)


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    img1 = np.zeros([1024, 1024, 3], dtype=np.uint8)
    img2 = np.zeros([1024, 1024, 3], dtype=np.uint8)
    img1 = img1[:H, :W, :]
    img2 = img2[:H, :W, :]
    img1 = img1.transpose([2, 0, 1])[None, ...]
    img2 = img2.transpose([2, 0, 1])[None, ...]
    img1s = (img1.astype(float) - 127.5) / 79.6875
    img2s = (img2.astype(float) - 127.5) / 79.6875
    segs = np.zeros([N, 1, H, W], dtype=np.uint8)
    segs[:, :, :100, :100] = 1
    img_metas = [
        {
            "img_shape": (H, W, C),
            "ori_shape": (H, W, C),
            "pad_shape": (H, W, C),
            "filename": "<demo>.png",
            "scale_factor": 1.0,
            "flip": False,
        }
        for _ in range(N)
    ]
    mm_inputs = {
        "img1s": torch.FloatTensor(img1s).requires_grad_(True),
        "img2s": torch.FloatTensor(img2s).requires_grad_(True),
        "img_metas": img_metas,
        "gt_semantic_seg": torch.LongTensor(segs),
    }
    return mm_inputs


def _prepare_input_img(img_path, test_pipeline, shape=None, rescale_shape=None):
    # build the data pipeline
    if shape is not None:
        test_pipeline[1]["img_scale"] = (shape[1], shape[0])
    test_pipeline[1]["transforms"][0]["keep_ratio"] = False
    test_pipeline = [LoadImage()] + test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img_path)
    data = test_pipeline(data)
    imgs = data["img"]
    img_metas = [i.data for i in data["img_metas"]]

    if rescale_shape is not None:
        for img_meta in img_metas:
            img_meta["ori_shape"] = tuple(rescale_shape) + (3,)

    mm_inputs = {"imgs": imgs, "img_metas": img_metas}

    return mm_inputs


def _update_input_img(img_list, img_meta_list, update_ori_shape=False):
    # update img and its meta list
    N, C, H, W = img_list[0].shape
    img_meta = img_meta_list[0][0]
    img_shape = (H, W, C)
    if update_ori_shape:
        ori_shape = img_shape
    else:
        ori_shape = img_meta["ori_shape"]
    pad_shape = img_shape
    new_img_meta_list = [
        [
            {
                "img_shape": img_shape,
                "ori_shape": ori_shape,
                "pad_shape": pad_shape,
                "filename": img_meta["filename"],
                "scale_factor": (
                    img_shape[1] / ori_shape[1],
                    img_shape[0] / ori_shape[0],
                )
                * 2,
                "flip": False,
            }
            for _ in range(N)
        ]
    ]

    return img_list, new_img_meta_list


def pytorch2onnx(
    model,
    mm_inputs,
    opset_version=11,
    show=False,
    output_file="tmp.onnx",
    verify=False,
    dynamic_export=False,
):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        mm_inputs (dict): Contain the input tensors and img_metas information.
        opset_version (int): The onnx op version. Default: 11.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
            Default: False.
        dynamic_export (bool): Whether to export ONNX with dynamic axis.
            Default: False.
    """
    model.cpu().eval()
    test_mode = model.test_cfg.mode

    if isinstance(model.decode_head, nn.ModuleList):
        num_classes = model.decode_head[-1].num_classes
    else:
        num_classes = model.decode_head.num_classes

    img1s = mm_inputs.pop("img1s")
    img2s = mm_inputs.pop("img2s")
    img_metas = mm_inputs.pop("img_metas")

    img1_list = [img[None, :] for img in img1s]
    img2_list = [img[None, :] for img in img2s]
    img_meta_list = [[img_meta] for img_meta in img_metas]
    # update img_meta
    img1_list, img_meta_list = _update_input_img(img1_list, img_meta_list)

    # replace original forward function
    origin_forward = model.forward
    dynamic_axes = None
    if dynamic_export:
        if test_mode == "slide":
            dynamic_axes = {"input": {0: "batch"}, "output": {1: "batch"}}
        else:
            dynamic_axes = {
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {1: "batch", 2: "height", 3: "width"},
            }

    register_extra_symbolics(opset_version)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (img1_list, img2_list, img_meta_list, False, dict(rescale=True)),
            output_file,
            input_names=["img1", "img2"],
            output_names=["output"],
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=show,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
        )
        print(f"Successfully exported ONNX model: {output_file}")
    os.system("python3 -m onnxsim {} {}".format(output_file, output_file))
    model.forward = origin_forward

    if verify:
        # check by onnx
        import onnx

        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        if dynamic_export and test_mode == "whole":
            # scale image for dynamic shape test
            img1_list = [
                nn.functional.interpolate(_, scale_factor=1.5) for _ in img1_list
            ]
            img2_list = [
                nn.functional.interpolate(_, scale_factor=1.5) for _ in img2_list
            ]
            # concate flip image for batch test
            flip_img1_list = [_.flip(-1) for _ in img1_list]
            img1_list = [
                torch.cat((ori_img, flip_img), 0)
                for ori_img, flip_img in zip(img1_list, flip_img1_list)
            ]
            flip_img2_list = [_.flip(-1) for _ in img2_list]
            img2_list = [
                torch.cat((ori_img, flip_img), 0)
                for ori_img, flip_img in zip(img2_list, flip_img2_list)
            ]
            # update img_meta
            img1_list, img_meta_list = _update_input_img(
                img1_list, img_meta_list, test_mode == "whole"
            )

        # check the numerical value
        # get pytorch output
        with torch.no_grad():
            pytorch_result = model(
                img1_list, img2_list, img_meta_list, return_loss=False
            )
            pytorch_result = np.stack(pytorch_result, 0)

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [node.name for node in onnx_model.graph.initializer]
        net_feed_input = sorted(list(set(input_all) - set(input_initializer)))
        assert len(net_feed_input) == 2
        sess = rt.InferenceSession(output_file)
        print(net_feed_input)
        onnx_result = sess.run(
            None,
            {
                net_feed_input[0]: img1_list[0].detach().numpy(),
                net_feed_input[1]: img2_list[0].detach().numpy(),
            },
        )[0][0]

        # show segmentation results
        if show:
            import cv2
            import os.path as osp

            img = img_meta_list[0][0]["filename"]
            if not osp.exists(img):
                img = img1s[0][:3, ...].permute(1, 2, 0) * 255
                img = img.detach().numpy().astype(np.uint8)
                ori_shape = img.shape[:2]
            else:
                ori_shape = LoadImage()({"img": img})["ori_shape"]

            # resize onnx_result to ori_shape
            onnx_result_ = cv2.resize(
                onnx_result[0].astype(np.uint8), (ori_shape[1], ori_shape[0])
            )
            show_result_pyplot(
                model,
                img,
                (onnx_result_,),
                palette=model.PALETTE,
                block=False,
                title="ONNXRuntime",
                opacity=0.5,
            )

            # resize pytorch_result to ori_shape
            pytorch_result_ = cv2.resize(
                pytorch_result[0].astype(np.uint8), (ori_shape[1], ori_shape[0])
            )
            show_result_pyplot(
                model,
                img,
                (pytorch_result_,),
                title="PyTorch",
                palette=model.PALETTE,
                opacity=0.5,
            )

        pytorch_result = pytorch_result[0, 0]
        pytorch_result = (pytorch_result * 255).astype(np.uint8)
        onnx_result = onnx_result[0]
        onnx_result = (onnx_result * 255).astype(np.uint8)
        from skimage import io

        io.imsave("pytorch_result.png", pytorch_result)
        io.imsave("onnx_result.png", onnx_result)
        np.testing.assert_allclose(
            pytorch_result.astype(np.float32)[0] / num_classes,
            onnx_result.astype(np.float32) / num_classes,
            rtol=1e-5,
            atol=1e-5,
            err_msg="The outputs are different between Pytorch and ONNX",
        )
        print("The outputs are same between Pytorch and ONNX")


def to_onnx(cfg, output_path, checkpoint_path=None, shape=(1024, 1024)):
    if len(shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(shape)
    else:
        raise ValueError("invalid input shape")
    cfg = copy(cfg)
    cfg.model.pretrained = None
    test_mode = cfg.model.test_cfg.mode

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    segmentor = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    # convert SyncBN to BN
    segmentor = _convert_batchnorm(segmentor)

    if checkpoint_path != None:
        checkpoint = load_checkpoint(segmentor, checkpoint_path, map_location="cpu")
        # segmentor.CLASSES = checkpoint['meta']['CLASSES']
        # segmentor.PALETTE = checkpoint['meta']['PALETTE']

    if isinstance(segmentor.decode_head, nn.ModuleList):
        num_classes = segmentor.decode_head[-1].num_classes
    else:
        num_classes = segmentor.decode_head.num_classes
    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    # convert model to onnx file
    pytorch2onnx(
        segmentor,
        mm_inputs,
        output_file=output_path,
    )
