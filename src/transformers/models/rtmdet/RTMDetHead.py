from functools import partial
from typing import Any, List, Optional, Sequence, Tuple, Type, Union
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn

from .sampler import PseudoSampler
from .batch_dsl_assigner import BatchDynamicSoftLabelAssigner
from .bbox_overlaps import bbox_overlaps
from .utils import get_dist_info, gt_instances_preprocess, reduce_mean
from .giou_loss import GIoULoss
from .focal_loss import QualityFocalLoss
from .modeling_rtmdet_cspnext import ConvModule

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from torch.nn.modules.utils import _pair
from abc import ABCMeta

DeviceType = Union[str, torch.device]
InstanceData = Any

def is_seq_of(seq: Any,
              expected_type: Union[Type, tuple],
              seq_type: Type = None) -> bool:
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True

def is_tuple_of(seq, expected_type):
    """Check whether it is a tuple of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)

def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def is_norm(layer: nn.Module,
            exclude: Union[type, tuple, None] = None) -> bool:
    """Check if a layer is a normalization layer.

    Args:
        layer (nn.Module): The layer to be checked.
        exclude (type | tuple[type]): Types to be excluded.

    Returns:
        bool: Whether the layer is a norm layer.
    """
    if exclude is not None:
        if not isinstance(exclude, tuple):
            exclude = (exclude, )
        if not is_tuple_of(exclude, type):
            raise TypeError(
                f'"exclude" must be either None or type or a tuple of types, '
                f'but got {type(exclude)}: {exclude}')

    if exclude and isinstance(layer, exclude):
        return False

    all_norm_bases = (_BatchNorm, _InstanceNorm, nn.GroupNorm, nn.LayerNorm)
    return isinstance(layer, all_norm_bases)


class RTMDetSepBNHeadModule(nn.Module):
    """Detection Head of RTMDet.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_base_priors (int): The number of priors (points) at a point
            on the feature grid.  Defaults to 1.
        feat_channels (int): Number of hidden channels. Used in child classes.
            Defaults to 256
        stacked_convs (int): Number of stacking convs of the head.
            Defaults to 2.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to (8, 16, 32).
        share_conv (bool): Whether to share conv layers between stages.
            Defaults to True.
        pred_kernel_size (int): Kernel size of ``nn.Conv2d``. Defaults to 1.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to ``dict(type='BN')``.
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Default: dict(type='SiLU', inplace=True).
    """
    def __init__(self,
            num_classes: int, 
            in_channels: int,
            widen_factor: float = 1.0,
            num_base_priors: int = 1,
            feat_channels: int = 256,
            stacked_convs: int = 2,
            featmap_strides: Sequence[int] = [8, 16, 32],
            share_conv: bool = True,
            pred_kernel_size: int = 1,
            conv_cfg: dict = None,
            norm_cfg: dict = dict(type='BN'),
            act_cfg: dict = dict(type='SiLU', inplace=True),
    ):
        super().__init__()
        self.share_conv = share_conv
        self.num_classes = num_classes
        self.pred_kernel_size = pred_kernel_size
        self.feat_channels = int(feat_channels * widen_factor)
        self.stacked_convs = stacked_convs
        self.num_base_priors = num_base_priors

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.featmap_strides = featmap_strides

        self.in_channels = int(in_channels * widen_factor)

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        for n in range(len(self.featmap_strides)):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.num_classes,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            self.rtm_reg.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * 4,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))

        if self.share_conv:
            for n in range(len(self.featmap_strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        # Use prior in model initialization to improve stability
        super().init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
        bias_cls = bias_init_with_prob(0.01)
        for rtm_cls, rtm_reg in zip(self.rtm_cls, self.rtm_reg):
            normal_init(rtm_cls, std=0.01, bias=bias_cls)
            normal_init(rtm_reg, std=0.01)

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
        """

        cls_scores = []
        bbox_preds = []
        for idx, x in enumerate(feats):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)

            reg_dist = self.rtm_reg[idx](reg_feat)
            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
        return tuple(cls_scores), tuple(bbox_preds)

class MlvlPointGenerator:
    """Standard points generator for multi-level (Mlvl) feature maps in 2D
    points-based detectors.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        offset (float): The offset of points, the value is normalized with
            corresponding stride. Defaults to 0.5.
    """

    def __init__(self,
                 strides: Union[List[int], List[Tuple[int, int]]],
                 offset: float = 0.5) -> None:
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self) -> int:
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    @property
    def num_base_priors(self) -> List[int]:
        """list[int]: The number of priors (points) at a point
        on the feature grid"""
        return [1 for _ in range(len(self.strides))]

    def _meshgrid(self,
                  x: Tensor,
                  y: Tensor,
                  row_major: bool = True) -> Tuple[Tensor, Tensor]:
        yy, xx = torch.meshgrid(y, x)
        if row_major:
            # warning .flatten() would cause error in ONNX exporting
            # have to use reshape here
            return xx.reshape(-1), yy.reshape(-1)

        else:
            return yy.reshape(-1), xx.reshape(-1)

    def grid_priors(self,
                    featmap_sizes: List[Tuple],
                    dtype: torch.dtype = torch.float32,
                    device: DeviceType = 'cuda',
                    with_stride: bool = False) -> List[Tensor]:
        """Generate grid points of multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            dtype (:obj:`dtype`): Dtype of priors. Defaults to torch.float32.
            device (str | torch.device): The device where the anchors will be
                put on.
            with_stride (bool): Whether to concatenate the stride to
                the last dimension of points.

        Return:
            list[torch.Tensor]: Points of  multiple feature levels.
            The sizes of each tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """

        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(
                featmap_sizes[i],
                level_idx=i,
                dtype=dtype,
                device=device,
                with_stride=with_stride)
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(self,
                                 featmap_size: Tuple[int],
                                 level_idx: int,
                                 dtype: torch.dtype = torch.float32,
                                 device: DeviceType = 'cuda',
                                 with_stride: bool = False) -> Tensor:
        """Generate grid Points of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps, arrange as
                (h, w).
            level_idx (int): The index of corresponding feature map level.
            dtype (:obj:`dtype`): Dtype of priors. Defaults to torch.float32.
            device (str | torch.device): The device the tensor will be put on.
                Defaults to 'cuda'.
            with_stride (bool): Concatenate the stride to the last dimension
                of points.

        Return:
            Tensor: Points of single feature levels.
            The shape of tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w, device=device) +
                   self.offset) * stride_w
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_x = shift_x.to(dtype)

        shift_y = (torch.arange(0, feat_h, device=device) +
                   self.offset) * stride_h
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_y = shift_y.to(dtype)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            # use `shape[0]` instead of `len(shift_xx)` for ONNX export
            stride_w = shift_xx.new_full((shift_xx.shape[0], ),
                                         stride_w).to(dtype)
            stride_h = shift_xx.new_full((shift_yy.shape[0], ),
                                         stride_h).to(dtype)
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h],
                                 dim=-1)
        all_points = shifts.to(device)
        return all_points

    def valid_flags(self,
                    featmap_sizes: List[Tuple[int, int]],
                    pad_shape: Tuple[int],
                    device: DeviceType = 'cuda') -> List[Tensor]:
        """Generate valid flags of points of multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            pad_shape (tuple(int)): The padded shape of the image,
                arrange as (h, w).
            device (str | torch.device): The device where the anchors will be
                put on.

        Return:
            list(torch.Tensor): Valid flags of points of multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            point_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / point_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size: Tuple[int, int],
                                 valid_size: Tuple[int, int],
                                 device = 'cuda') -> Tensor:
        """Generate the valid flags of points of a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps, arrange as
                as (h, w).
            valid_size (tuple[int]): The valid size of the feature maps.
                The size arrange as as (h, w).
            device (str | torch.device): The device where the flags will be
            put on. Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each points in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid

    def sparse_priors(self,
                      prior_idxs: Tensor,
                      featmap_size: Tuple[int],
                      level_idx: int,
                      dtype: torch.dtype = torch.float32,
                      device: DeviceType = 'cuda') -> Tensor:
        """Generate sparse points according to the ``prior_idxs``.

        Args:
            prior_idxs (Tensor): The index of corresponding anchors
                in the feature map.
            featmap_size (tuple[int]): feature map size arrange as (w, h).
            level_idx (int): The level index of corresponding feature
                map.
            dtype (obj:`torch.dtype`): Date type of points. Defaults to
                ``torch.float32``.
            device (str | torch.device): The device where the points is
                located.
        Returns:
            Tensor: Anchor with shape (N, 2), N should be equal to
            the length of ``prior_idxs``. And last dimension
            2 represent (coord_x, coord_y).
        """
        height, width = featmap_size
        x = (prior_idxs % width + self.offset) * self.strides[level_idx][0]
        y = ((prior_idxs // width) % height +
             self.offset) * self.strides[level_idx][1]
        prioris = torch.stack([x, y], 1).to(dtype)
        prioris = prioris.to(device)
        return prioris

def distance2bbox(
    points: Tensor,
    distance: Tensor,
    max_shape: Optional[Union[Sequence[int], Tensor,
                              Sequence[Sequence[int]]]] = None
) -> Tensor:
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
        max_shape (Union[Sequence[int], Tensor, Sequence[Sequence[int]]],
            optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.

    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """

    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
            # speed up
            bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
            return bboxes

        # clip bboxes with dynamic `min` and `max` for onnx
        if torch.onnx.is_in_onnx_export():
            # TODO: delete
            #from mmdet.core.export import dynamic_clip_for_onnx
            raise NotImplementedError('dynamic_clip_for_onnx')
            x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape)
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
            return bboxes
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)

        min_xy = x1.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape],
                           dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes


def bbox2distance(points: Tensor,
                  bbox: Tensor,
                  max_dis: Optional[float] = None,
                  eps: float = 0.1) -> Tensor:
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2) or (b, n, 2), [x, y].
        bbox (Tensor): Shape (n, 4) or (b, n, 4), "xyxy" format
        max_dis (float, optional): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """
    left = points[..., 0] - bbox[..., 0]
    top = points[..., 1] - bbox[..., 1]
    right = bbox[..., 2] - points[..., 0]
    bottom = bbox[..., 3] - points[..., 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


class DistancePointBBoxCoder(metaclass=ABCMeta):
    """Distance Point BBox coder.

    This coder encodes gt bboxes (x1, y1, x2, y2) into (top, bottom, left,
    right) and decode it back to the original.
    """

    encode_size = 4

    def __init__(self, 
                 clip_border: Optional[bool] = True,
                 use_box_type: bool = False
        ) -> None:
        super().__init__()
        self.clip_border = clip_border
        self.use_box_type = use_box_type

    def decode(
        self,
        points: torch.Tensor,
        pred_bboxes: torch.Tensor,
        stride: torch.Tensor,
        max_shape: Optional[Union[Sequence[int], torch.Tensor,
                                  Sequence[Sequence[int]]]] = None
    ) -> torch.Tensor:
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (B, N, 2) or (N, 2).
            pred_bboxes (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom). Shape (B, N, 4)
                or (N, 4)
            stride (Tensor): Featmap stride.
            max_shape (Sequence[int] or torch.Tensor or Sequence[
                Sequence[int]],optional): Maximum bounds for boxes, specifies
                (H, W, C) or (H, W). If priors shape is (B, N, 4), then
                the max_shape should be a Sequence[Sequence[int]],
                and the length of max_shape should also be B.
                Default None.
        Returns:
            Tensor: Boxes with shape (N, 4) or (B, N, 4)
        """
        assert points.size(-2) == pred_bboxes.size(-2)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == 4
        if self.clip_border is False:
            max_shape = None

        pred_bboxes = pred_bboxes * stride[None, :, None]

        return distance2bbox(points, pred_bboxes, max_shape)

    def encode(self,
               points: torch.Tensor,
               gt_bboxes: torch.Tensor,
               max_dis: float = 16.,
               eps: float = 0.01) -> torch.Tensor:
        """Encode bounding box to distances. The rewrite is to support batch
        operations.

        Args:
            points (Tensor): Shape (B, N, 2) or (N, 2), The format is [x, y].
            gt_bboxes (Tensor or :obj:`BaseBoxes`): Shape (N, 4), The format
                is "xyxy"
            max_dis (float): Upper bound of the distance. Default to 16..
            eps (float): a small value to ensure target < max_dis, instead <=.
                Default 0.01.

        Returns:
            Tensor: Box transformation deltas. The shape is (N, 4) or
             (B, N, 4).
        """

        assert points.size(-2) == gt_bboxes.size(-2)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 4
        return bbox2distance(points, gt_bboxes, max_dis, eps)

def get_prior_xy_info(index: int, num_base_priors: int,
                      featmap_sizes: int) -> Tuple[int, int, int]:
    """Get prior index and xy index in feature map by flatten index."""
    _, featmap_w = featmap_sizes
    priors = index % num_base_priors
    xy_index = index // num_base_priors
    grid_y = xy_index // featmap_w
    grid_x = xy_index % featmap_w
    return priors, grid_x, grid_y

class RTMDetHead(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 prior_match_thr: float = 4.0,
                 near_neighbor_thr: float = 0.5,
                 ignore_iof_thr: float = -1.0,
                 obj_level_weights: List[float] = [4.0, 1.0, 0.4],
        ):
        super().__init__()

        featmap_strides: Sequence[int] = [8, 16, 32]

        self.num_classes = num_classes
        self.cls_out_channels = self.num_classes
        self.head_module = RTMDetSepBNHeadModule(
            num_classes=self.num_classes,
            in_channels=256,
            featmap_strides=featmap_strides,
        )
        self.prior_generator = MlvlPointGenerator(
            strides=featmap_strides,
            offset=0
        )
        self.box_coder = DistancePointBBoxCoder()
        self.loss_cls = QualityFocalLoss(use_sigmoid=True, beta=2.0, loss_weight=1.0)
        self.loss_bbox = GIoULoss(loss_weight=2.0)
        
        self.featmap_strides = self.head_module.featmap_strides
        self.num_levels = len(self.featmap_strides)

        self.prior_match_thr = prior_match_thr
        self.near_neighbor_thr = near_neighbor_thr
        self.ignore_iof_thr = ignore_iof_thr

        self.featmap_sizes = [torch.empty(1)] * self.num_levels
        self.obj_level_weights = obj_level_weights

        self.num_base_priors = self.prior_generator.num_base_priors[0]

        #  train_cfg=dict(
        # assigner=dict(
        #     type='BatchDynamicSoftLabelAssigner',
        #     num_classes=num_classes,
        #     topk=dsl_topk,
        #     iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
        # allowed_border=-1,
        # pos_weight=-1,
        # debug=False),

        self.assigner = BatchDynamicSoftLabelAssigner(num_classes=num_classes)
        # if self.train_cfg.get('sampler', None) is not None:
        #     self.sampler = TASK_UTILS.build(
        #         self.train_cfg.sampler, default_args=dict(context=self))
        # else:
        self.sampler = PseudoSampler(context=self)

        self.featmap_sizes_train = None
        self.flatten_priors_train = None

    def init_weights(self) -> None:
        """Initialize the weights."""
        #super().init_weights()
        # avoid init_cfg overwrite the initialization of `conv_offset`
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        return self.head_module(x)

    def loss(self, x: Tuple[Tensor], batch_data_samples: Union[list,
                                                               dict]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`], dict): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """

        if isinstance(batch_data_samples, list):
            losses = super().loss(x, batch_data_samples)
        else:
            outs = self(x)
            # Fast version
            loss_inputs = outs + (batch_data_samples['bboxes_labels'],
                                  batch_data_samples['img_metas'])
            losses = self.loss_by_feat(*loss_inputs)

        return losses

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore = None) -> dict:
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Decoded box for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [tl_x, tl_y, br_x, br_y] format.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        device = cls_scores[0].device

        # If the shape does not equal, generate new one
        if featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = featmap_sizes
            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                featmap_sizes, device=device, with_stride=True)
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)

        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1).contiguous()

        flatten_bboxes = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ], 1)
        flatten_bboxes = flatten_bboxes * self.flatten_priors_train[..., -1,
                                                                    None]
        flatten_bboxes = distance2bbox(self.flatten_priors_train[..., :2],
                                       flatten_bboxes)

        assigned_result = self.assigner(flatten_bboxes.detach(),
                                        flatten_cls_scores.detach(),
                                        self.flatten_priors_train, gt_labels,
                                        gt_bboxes, pad_bbox_flag)

        labels = assigned_result['assigned_labels'].reshape(-1)
        label_weights = assigned_result['assigned_labels_weights'].reshape(-1)
        bbox_targets = assigned_result['assigned_bboxes'].reshape(-1, 4)
        assign_metrics = assigned_result['assign_metrics'].reshape(-1)
        cls_preds = flatten_cls_scores.reshape(-1, self.num_classes)
        bbox_preds = flatten_bboxes.reshape(-1, 4)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        avg_factor = reduce_mean(assign_metrics.sum()).clamp_(min=1).item()

        loss_cls = self.loss_cls(
            cls_preds, (labels, assign_metrics),
            label_weights,
            avg_factor=avg_factor)

        if len(pos_inds) > 0:
            loss_bbox = self.loss_bbox(
                bbox_preds[pos_inds],
                bbox_targets[pos_inds],
                weight=assign_metrics[pos_inds],
                avg_factor=avg_factor)
        else:
            loss_bbox = bbox_preds.sum() * 0

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    def _convert_gt_to_norm_format(self,
                                   batch_gt_instances: Sequence[InstanceData],
                                   batch_img_metas: Sequence[dict]) -> Tensor:
        if isinstance(batch_gt_instances, torch.Tensor):
            # fast version
            img_shape = batch_img_metas[0]['batch_input_shape']
            gt_bboxes_xyxy = batch_gt_instances[:, 2:]
            xy1, xy2 = gt_bboxes_xyxy.split((2, 2), dim=-1)
            gt_bboxes_xywh = torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)
            gt_bboxes_xywh[:, 1::2] /= img_shape[0]
            gt_bboxes_xywh[:, 0::2] /= img_shape[1]
            batch_gt_instances[:, 2:] = gt_bboxes_xywh

            # (num_base_priors, num_bboxes, 6)
            batch_targets_normed = batch_gt_instances.repeat(
                self.num_base_priors, 1, 1)
        else:
            batch_target_list = []
            # Convert xyxy bbox to yolo format.
            for i, gt_instances in enumerate(batch_gt_instances):
                img_shape = batch_img_metas[i]['batch_input_shape']
                bboxes = gt_instances.bboxes
                labels = gt_instances.labels

                xy1, xy2 = bboxes.split((2, 2), dim=-1)
                bboxes = torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)
                # normalized to 0-1
                bboxes[:, 1::2] /= img_shape[0]
                bboxes[:, 0::2] /= img_shape[1]

                index = bboxes.new_full((len(bboxes), 1), i)
                # (batch_idx, label, normed_bbox)
                target = torch.cat((index, labels[:, None].float(), bboxes),
                                   dim=1)
                batch_target_list.append(target)

            # (num_base_priors, num_bboxes, 6)
            batch_targets_normed = torch.cat(
                batch_target_list, dim=0).repeat(self.num_base_priors, 1, 1)

        # (num_base_priors, num_bboxes, 1)
        batch_targets_prior_inds = self.prior_inds.repeat(
            1, batch_targets_normed.shape[1])[..., None]
        # (num_base_priors, num_bboxes, 7)
        # (img_ind, labels, bbox_cx, bbox_cy, bbox_w, bbox_h, prior_ind)
        batch_targets_normed = torch.cat(
            (batch_targets_normed, batch_targets_prior_inds), 2)
        return batch_targets_normed

    def _decode_bbox_to_xywh(self, bbox_pred, priors_base_sizes) -> Tensor:
        bbox_pred = bbox_pred.sigmoid()
        pred_xy = bbox_pred[:, :2] * 2 - 0.5
        pred_wh = (bbox_pred[:, 2:] * 2)**2 * priors_base_sizes
        decoded_bbox_pred = torch.cat((pred_xy, pred_wh), dim=-1)
        return decoded_bbox_pred

    def _loss_by_feat_with_ignore(
            self, cls_scores: Sequence[Tensor], bbox_preds: Sequence[Tensor],
            objectnesses: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: Sequence[Tensor]) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (Sequence[Tensor]): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_gt_instances (Sequence[InstanceData]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (Sequence[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (Sequence[Tensor]): Ignore boxes with
                batch_ids and labels, each is a 2D-tensor, the channel number
                is 6, means that (batch_id, label, xmin, ymin, xmax, ymax).
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        # 1. Convert gt to norm format
        batch_targets_normed = self._convert_gt_to_norm_format(
            batch_gt_instances, batch_img_metas)

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes

        device = cls_scores[0].device
        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        scaled_factor = torch.ones(7, device=device)

        for i in range(self.num_levels):
            batch_size, _, h, w = bbox_preds[i].shape
            target_obj = torch.zeros_like(objectnesses[i])

            not_ignore_flags = bbox_preds[i].new_ones(batch_size,
                                                      self.num_base_priors, h,
                                                      w)

            ignore_overlaps = bbox_overlaps(self.mlvl_priors[i],
                                            batch_gt_instances_ignore[..., 2:],
                                            'iof')
            ignore_max_overlaps, ignore_max_ignore_index = ignore_overlaps.max(
                dim=1)

            batch_inds = batch_gt_instances_ignore[:,
                                                   0][ignore_max_ignore_index]
            ignore_inds = (ignore_max_overlaps > self.ignore_iof_thr).nonzero(
                as_tuple=True)[0]
            batch_inds = batch_inds[ignore_inds].long()
            ignore_priors, ignore_grid_xs, ignore_grid_ys = get_prior_xy_info(
                ignore_inds, self.num_base_priors, self.featmap_sizes[i])
            not_ignore_flags[batch_inds, ignore_priors, ignore_grid_ys,
                             ignore_grid_xs] = 0

            # empty gt bboxes
            if batch_targets_normed.shape[1] == 0:
                loss_box += bbox_preds[i].sum() * 0
                loss_cls += cls_scores[i].sum() * 0
                loss_obj += self.loss_obj(
                    objectnesses[i],
                    target_obj,
                    weight=not_ignore_flags,
                    avg_factor=max(not_ignore_flags.sum(),
                                   1)) * self.obj_level_weights[i]
                continue

            priors_base_sizes_i = self.priors_base_sizes[i]
            # feature map scale whwh
            scaled_factor[2:6] = torch.tensor(
                bbox_preds[i].shape)[[3, 2, 3, 2]]
            # Scale batch_targets from range 0-1 to range 0-features_maps size.
            # (num_base_priors, num_bboxes, 7)
            batch_targets_scaled = batch_targets_normed * scaled_factor

            # 2. Shape match
            wh_ratio = batch_targets_scaled[...,
                                            4:6] / priors_base_sizes_i[:, None]
            match_inds = torch.max(
                wh_ratio, 1 / wh_ratio).max(2)[0] < self.prior_match_thr
            batch_targets_scaled = batch_targets_scaled[match_inds]

            # no gt bbox matches anchor
            if batch_targets_scaled.shape[0] == 0:
                loss_box += bbox_preds[i].sum() * 0
                loss_cls += cls_scores[i].sum() * 0
                loss_obj += self.loss_obj(
                    objectnesses[i],
                    target_obj,
                    weight=not_ignore_flags,
                    avg_factor=max(not_ignore_flags.sum(),
                                   1)) * self.obj_level_weights[i]
                continue

            # 3. Positive samples with additional neighbors

            # check the left, up, right, bottom sides of the
            # targets grid, and determine whether assigned
            # them as positive samples as well.
            batch_targets_cxcy = batch_targets_scaled[:, 2:4]
            grid_xy = scaled_factor[[2, 3]] - batch_targets_cxcy
            left, up = ((batch_targets_cxcy % 1 < self.near_neighbor_thr) &
                        (batch_targets_cxcy > 1)).T
            right, bottom = ((grid_xy % 1 < self.near_neighbor_thr) &
                             (grid_xy > 1)).T
            offset_inds = torch.stack(
                (torch.ones_like(left), left, up, right, bottom))

            batch_targets_scaled = batch_targets_scaled.repeat(
                (5, 1, 1))[offset_inds]
            retained_offsets = self.grid_offset.repeat(1, offset_inds.shape[1],
                                                       1)[offset_inds]

            # prepare pred results and positive sample indexes to
            # calculate class loss and bbox lo
            _chunk_targets = batch_targets_scaled.chunk(4, 1)
            img_class_inds, grid_xy, grid_wh, priors_inds = _chunk_targets
            priors_inds, (img_inds, class_inds) = priors_inds.long().view(
                -1), img_class_inds.long().T

            grid_xy_long = (grid_xy -
                            retained_offsets * self.near_neighbor_thr).long()
            grid_x_inds, grid_y_inds = grid_xy_long.T
            bboxes_targets = torch.cat((grid_xy - grid_xy_long, grid_wh), 1)

            # 4. Calculate loss
            # bbox loss
            retained_bbox_pred = bbox_preds[i].reshape(
                batch_size, self.num_base_priors, -1, h,
                w)[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]
            priors_base_sizes_i = priors_base_sizes_i[priors_inds]
            decoded_bbox_pred = self._decode_bbox_to_xywh(
                retained_bbox_pred, priors_base_sizes_i)

            not_ignore_weights = not_ignore_flags[img_inds, priors_inds,
                                                  grid_y_inds, grid_x_inds]
            loss_box_i, iou = self.loss_bbox(
                decoded_bbox_pred,
                bboxes_targets,
                weight=not_ignore_weights,
                avg_factor=max(not_ignore_weights.sum(), 1))
            loss_box += loss_box_i

            # obj loss
            iou = iou.detach().clamp(0)
            target_obj[img_inds, priors_inds, grid_y_inds,
                       grid_x_inds] = iou.type(target_obj.dtype)
            loss_obj += self.loss_obj(
                objectnesses[i],
                target_obj,
                weight=not_ignore_flags,
                avg_factor=max(not_ignore_flags.sum(),
                               1)) * self.obj_level_weights[i]

            # cls loss
            if self.num_classes > 1:
                pred_cls_scores = cls_scores[i].reshape(
                    batch_size, self.num_base_priors, -1, h,
                    w)[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]

                target_class = torch.full_like(pred_cls_scores, 0.)
                target_class[range(batch_targets_scaled.shape[0]),
                             class_inds] = 1.
                loss_cls += self.loss_cls(
                    pred_cls_scores,
                    target_class,
                    weight=not_ignore_weights[:, None].repeat(
                        1, self.num_classes),
                    avg_factor=max(not_ignore_weights.sum(), 1))
            else:
                loss_cls += cls_scores[i].sum() * 0

        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * batch_size * world_size,
            loss_obj=loss_obj * batch_size * world_size,
            loss_bbox=loss_box * batch_size * world_size)