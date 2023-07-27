# -*- conding: utf-8 -*-
from torch import nn

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from .change_detector import ChangedetEncoderDecoder


@SEGMENTORS.register_module()
class CascadeChangeDetector(ChangedetEncoderDecoder):
    """Cascade Encoder Decoder segmentors.

    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders of
    CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.
    """

    def __init__(
        self,
        num_stages,
        backbone,
        decode_head,
        neck=None,
        auxiliary_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
        **kwargs,
    ):
        """
        Args:
            num_stages (int): Number of stages in cascade.
            backbone (dict): Config dict for backbone.
            decode_head (list[dict]): Config dict for decode head.
            neck (dict): Config dict for neck.
            auxiliary_head (dict): Config dict for auxiliary head.
            train_cfg (dict): Training config of the model.
            test_cfg (dict): Testing config of the model.
            pretrained (str): Path to pre-trained model.
            init_cfg (dict): The Config for initialization.
            kwargs (dict): Other keyword arguments.
        """
        self.num_stages = num_stages
        super(CascadeChangeDetector, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            **kwargs,
        )

    def _init_decode_head(self, decode_head):
        """
        Initialize ``decode_head``.
        Args:
            decode_head (list[dict]): Config dict for decode head.
        """
        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        self.decode_head = nn.ModuleList()
        for i in range(self.num_stages):
            self.decode_head.append(builder.build_head(decode_head[i]))
        self.align_corners = self.decode_head[-1].align_corners
        self.num_classes = self.decode_head[-1].num_classes

    def encode_decode(self, img1, img2, img_metas):
        """
        Encode images with backbone and decode into a semantic segmentation
        map of the same size as input.
        Args:
            img1 (torch.Tensor): The first input image.
            img2 (torch.Tensor): The second input image.
            img_metas (list[dict]): Meta information of input images.
        Returns:
            torch.Tensor: The segmentation map of the same size as input.
        """
        x, x1, x2 = self.extract_feat(img1, img2)
        out = self.decode_head[0].forward_test(x, img_metas, self.test_cfg)
        for i in range(1, self.num_stages):
            out = self.decode_head[i].forward_test(x, out, img_metas, self.test_cfg)
        out = resize(
            input=out,
            size=img1.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """
        Run forward function and calculate loss for decode head in training.
        Args:
            x (torch.Tensor): The input tensor.
            img_metas (list[dict]): Meta information of input images.
            gt_semantic_seg (torch.Tensor): Ground truth segmentation maps.
        Returns:
            dict: A dict containing the loss for decode head.
        """
        losses = dict()

        loss_decode = self.decode_head[0].forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg
        )

        losses.update(add_prefix(loss_decode, "decode_0"))

        for i in range(1, self.num_stages):
            # forward test again, maybe unnecessary for most methods.
            prev_outputs = self.decode_head[i - 1].forward_test(
                x, img_metas, self.test_cfg
            )
            loss_decode = self.decode_head[i].forward_train(
                x, prev_outputs, img_metas, gt_semantic_seg, self.train_cfg
            )
            losses.update(add_prefix(loss_decode, f"decode_{i}"))

        return losses
