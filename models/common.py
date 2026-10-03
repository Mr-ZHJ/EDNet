# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import json
import math
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.cuda import amp

# --------------------添加声明---------------------------------
import torch.nn.functional as F
# --------------------添加声明---------------------------------

from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

# class Conv(nn.Module):
#     # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
#     default_act = nn.SiLU()  # default activation

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         """Initializes a standard convolution layer with optional batch normalization and activation."""
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         """Applies a fused convolution and activation function to the input tensor `x`."""
#         return self.act(self.conv(x))
    
#-------------------------------Conv_RepBN start------------------------------
class RepBN2d(nn.Module):
    def __init__(self, channels):
        super(RepBN2d, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm2d(channels)  # 使用BatchNorm2d

    def forward(self, x):
        # BatchNorm2d处理四维输入数据
        x = self.bn(x) + self.alpha * x  # BatchNorm + alpha * input
        return x
    
class Conv_RepBN(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = RepBN2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
#-------------------------------Conv_RepBN end-------------------------------------------------------------------

class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))

class Bottleneck_C2f(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_C2f(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)
    


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=None, dnn=False, data=None):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx with --dnn
        #   OpenVINO:                       *.xml
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs = self.model_type(w)  # get backend
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        w = attempt_download(w)  # download if not local
        if data:  # data.yaml path (optional)
            with open(data, errors='ignore') as f:
                names = yaml.safe_load(f)['names']  # class names

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, map_location=device)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files)
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])  # extra_files dict
                stride, names = int(d['stride']), d['names']
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available()
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
        elif xml:  # OpenVINO
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements(('openvino-dev',))  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            import openvino.inference_engine as ie
            core = ie.IECore()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir
            network = core.read_network(model=w, weights=Path(w).with_suffix('.bin'))  # *.xml, *.bin paths
            executable_network = core.load_network(network, device_name='CPU', num_requests=1)
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            bindings = OrderedDict()
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = tuple(model.get_binding_shape(index))
                data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
                bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            context = model.create_execution_context()
            batch_size = bindings['images'].shape[0]
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            if saved_model:  # SavedModel
                LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
                import tensorflow as tf
                keras = False  # assume TF1 saved_model
                model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
            elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
                import tensorflow as tf

                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    ge = x.graph.as_graph_element
                    return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

                gd = tf.Graph().as_graph_def()  # graph_def
                gd.ParseFromString(open(w, 'rb').read())
                frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs="Identity:0")
            elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                    from tflite_runtime.interpreter import Interpreter, load_delegate
                except ImportError:
                    import tensorflow as tf
                    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
                if edgetpu:  # Edge TPU https://coral.ai/software/#edgetpu-runtime
                    LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                    delegate = {'Linux': 'libedgetpu.so.1',
                                'Darwin': 'libedgetpu.1.dylib',
                                'Windows': 'edgetpu.dll'}[platform.system()]
                    interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
                else:  # Lite
                    LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                    interpreter = Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
            elif tfjs:
                raise Exception('ERROR: YOLOv5 TF.js inference is not supported')
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.pt or self.jit:  # PyTorch
            y = self.model(im) if self.jit else self.model(im, augment=augment, visualize=visualize)
            return y if val else y[0]
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            desc = self.ie.TensorDesc(precision='FP32', dims=im.shape, layout='NCHW')  # Tensor Description
            request = self.executable_network.requests[0]  # inference request
            request.set_blob(blob_name='images', blob=self.ie.Blob(desc, im))  # name=next(iter(request.input_blobs))
            request.infer()
            y = request.output_blobs['output'].buffer  # name=next(iter(request.output_blobs))
        elif self.engine:  # TensorRT
            assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = self.bindings['output'].data
        elif self.coreml:  # CoreML
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                k = 'var_' + str(sorted(int(k.replace('var_', '')) for k in y)[-1])  # output key
                y = y[k]  # output
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            if self.saved_model:  # SavedModel
                y = (self.model(im, training=False) if self.keras else self.model(im)[0]).numpy()
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im)).numpy()
            else:  # Lite or Edge TPU
                input, output = self.input_details[0], self.output_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = self.interpreter.get_tensor(output['index'])
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
            y[..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        y = torch.tensor(y) if isinstance(y, np.ndarray) else y
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640), half=False):
        # Warmup model by running inference once
        if self.pt or self.jit or self.onnx or self.engine:  # warmup types
            if isinstance(self.device, torch.device) and self.device.type != 'cpu':  # only warmup GPU models
                im = torch.zeros(*imgsz).to(self.device).type(torch.half if half else torch.float)  # input image
                self.forward(im)  # warmup

    @staticmethod
    def model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        from export import export_formats
        suffixes = list(export_formats().Suffix) + ['.xml']  # export suffixes
        check_suffix(p, suffixes)  # checks
        p = Path(p).name  # eliminate trailing separators
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2 = (s in p for s in suffixes)
        xml |= xml2  # *_openvino_model or *.xml
        tflite &= not edgetpu  # *.tflite
        return pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model):
        super().__init__()
        LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters()) if self.pt else torch.zeros(1)  # for device and type
        autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=autocast):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, self.stride) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1 if self.pt else size, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=autocast):
            # Inference
            y = self.model(x, augment, profile)  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y if self.dmb else y[0], self.conf, iou_thres=self.iou, classes=self.classes,
                                    agnostic=self.agnostic, multi_label=self.multi_label, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=(0, 0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.imgs[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


# ---------------------------------EfficientNetV1 start----------------------------------------------
# SE-Net Adaptive avg pooling --> fc --> fc --> Sigmoid
# SeBlock:实现 Squeeze-and-Excitation（SE）模块，用于自适应地调整通道特征的重要性。
class SeBlock(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super().__init__()
        self.Squeeze = nn.AdaptiveAvgPool2d(1)

        self.Excitation = nn.Sequential()
        self.Excitation.add_module('FC1', nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1))  # 1*1卷积与此效果相同
        self.Excitation.add_module('ReLU', nn.ReLU())
        self.Excitation.add_module('FC2', nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1))
        self.Excitation.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        y = self.Squeeze(x)
        ouput = self.Excitation(y)
        return x * (ouput.expand_as(x))


# drop_connect:实现随机丢弃连接，以减少模型复杂度并提高泛化能力。
class drop_connect:
    def __init__(self, drop_connect_rate):
        self.drop_connect_rate = drop_connect_rate

    def forward(self, x, training):
        if not training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.shape[0]
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)  # 1
        x = (x / keep_prob) * binary_mask
        return x



# EfficientNet的Stem模块，用于输入图像的预处理和特征提取。
class stem(nn.Module):
    def __init__(self, c1, c2, act='ReLU6'):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=c2)
        if act == 'ReLU6':
            self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# stem模块优化1，使用 Kaiming 初始化卷积层权重，并使用 Swish 激活函数。
class optstem_v1(nn.Module):
    def __init__(self, c1, c2, act='Swish', dropout_rate=0.1):
        super().__init__()
        # 卷积层
        self.conv = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)
        # 使用 Kaiming 初始化卷积层权重
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        # 批归一化层
        self.bn = nn.BatchNorm2d(num_features=c2)
        # # Dropout 层
        # self.dropout = drop_connect(dropout_rate)
        # 激活函数
        if act == 'Mish':
            self.act = nn.Mish(inplace=True)
        elif act == 'Swish':
            self.act = nn.SiLU(inplace=True)
        elif act == 'ReLU6':
            self.act = nn.ReLU6(inplace=True)
        elif act == 'HardSwish':
            self.act = nn.Hardswish(inplace=True)    

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.dropout(x)
        x = self.act(x)
        return x

# stem模块优化2，使用多分支卷积，并添加 SE 注意力机制。
class optstem_v2(nn.Module):
    def __init__(self, c1, c2, act='ReLU6'):
        super().__init__()
        # 多分支卷积
        self.conv1 = nn.Conv2d(c1, c2 // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c1, c2 // 2, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features=c2)
        if act == 'SiLU':
            self.act = nn.SiLU(inplace=True)
        elif act == 'ReLU6':
            self.act = nn.ReLU6(inplace=True)
        # 添加 SE 注意力机制
        self.se = SeBlock(c2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        return x
    

# stem模块优化3，引入残差连接。
class optstem_v3(nn.Module):
    def __init__(self, c1, c2, act='ReLU6'):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=c2)
        if act == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif act == 'ReLU6':
            self.act = nn.ReLU6(inplace=True)
        if c1 != c2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(c2)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        shortcut = self.shortcut(x)
        return self.act(out + shortcut)

# MBConvBlock:实现MobileNetV2中的MBConv模块，用于特征提取和通道调整。
class MBConvBlock(nn.Module):
    def __init__(self, inp, final_oup, k, s, expand_ratio, drop_connect_rate, has_se=False):
        super(MBConvBlock, self).__init__()

        self._momentum = 0.01
        self._epsilon = 1e-3
        self.input_filters = inp
        self.output_filters = final_oup
        self.stride = s
        self.expand_ratio = expand_ratio
        self.has_se = has_se
        self.id_skip = True  # skip connection and drop connect
        se_ratio = 0.25

        # Expansion phase
        oup = inp * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._momentum, eps=self._epsilon)

        # Depthwise convolution phase
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, padding=(k - 1) // 2, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._momentum, eps=self._epsilon)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(inp * se_ratio))
            self.se = SeBlock(oup, 4)

        # Output phase
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._momentum, eps=self._epsilon)
        self._relu = nn.ReLU6(inplace=True)

        self.drop_connect = drop_connect(drop_connect_rate)

    def forward(self, x, drop_connect_rate=None):
        """
        :param x: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        identity = x
        if self.expand_ratio != 1:
            x = self._relu(self._bn0(self._expand_conv(x)))
        x = self._relu(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x = self.se(x)

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        if self.id_skip and self.stride == 1 and self.input_filters == self.output_filters:
            if drop_connect_rate:
                x = self.drop_connect(x, training=self.training)
            x += identity  # skip connection
        return x

# -------------------------------------EfficientNetV1 end------------------------------------------

# -----------------------------------EfficientNetV2 start-------------------------------------------
# 定义Drop Path操作，用于随机丢弃部分特征
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob

    shape = (x.shape[0],) + (1,) * (x.ndim - 1)

    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)

    random_tensor.floor_()  # binarize

    output = x.div(keep_prob) * random_tensor

    return output

# 定义DropPath层
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# 定义Squeeze-and-Excite模块
class SqueezeExcite_efficientv2(nn.Module):
    def __init__(self, c1, c2, se_ratio=0.25, act_layer=nn.ReLU):
        super().__init__()

        self.gate_fn = nn.Sigmoid()

        reduced_chs = int(c1 * se_ratio)

        # 全局平均池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 降维卷积层
        self.conv_reduce = nn.Conv2d(c1, reduced_chs, 1, bias=True)

        # ReLU激活函数
        self.act1 = act_layer(inplace=True)

        # 升维卷积层
        self.conv_expand = nn.Conv2d(reduced_chs, c2, 1, bias=True)

    def forward(self, x):
        # 全局平均池化
        x_se = self.avg_pool(x)
        # 降维卷积
        x_se = self.conv_reduce(x_se)
        # ReLU激活
        x_se = self.act1(x_se)
        # 升维卷积
        x_se = self.conv_expand(x_se)
        # Sigmoid激活
        x_se = self.gate_fn(x_se)
        # 将x_se维度扩展为和x一样的维度
        x = x * (x_se.expand_as(x))
        return x

# Fused-MBConv 将MBConv中的depthwise conv3×3和扩展conv1×1替换为单个常规conv3×3
class FusedMBConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1, se_ration=0, dropout_rate=0.2, drop_connect_rate=0.2):
        super().__init__()

        # shortcut 指的是到残差结构，expansion是为了先升维，再卷积，再降维，再残差
        self.has_shortcut = (s == 1 and c1 == c2)  # 只要是步长为1并且输入输出特征图大小相等，就是True就可以使用残差结构连接
        self.has_expansion = expansion != 1  # expansion不为1时，输出特征图维度就为expansion * c1，k倍的c1，扩展维度
        expanded_c = c1 * expansion

        if self.has_expansion:
            self.expansion_conv = stem(c1, expanded_c, kernel_size=k, stride=s)
            self.project_conv = stem(expanded_c, c2, kernel_size=1, stride=1)
        else:
            self.project_conv = stem(c1, c2, kernel_size=k, stride=s)

        self.drop_connect_rate = drop_connect_rate
        if self.has_shortcut and drop_connect_rate > 0:
            self.dropout = DropPath(drop_connect_rate)

    def forward(self, x):
        if self.has_expansion:
            result = self.expansion_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        if self.has_shortcut:
            if self.drop_connect_rate > 0:
                result = self.dropout(result)
            result += x

        return result
# -----------------------------------EfficientNetV2 end-------------------------------------------

# -----------------------------------SimAM start --------------------------
# SimAM:SimAM（Simple Attention Module）注意力模块，它是一种轻量级、无参数的注意力机制，
# 可用于卷积神经网络中，增强模型对特征图中重要信息的关注。
class SimAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
# -----------------------------------SimAM end --------------------------

# -----------------------------------OptSimAM start --------------------------
class OptSimAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(OptSimAM, self).__init__()
        self.activation = nn.Sigmoid() # default activation function
        # self.activation = nn.ReLU()  # ReLU
        # self.activation = nn.LeakyReLU(negative_slope=0.01)  # Leaky ReLU
        # self.activation = nn.Swish()  # Swish
        # self.activation = nn.Mish()  # Mish
        # self.activation = nn.ELU()  # ELU
        # self.activation = nn.GELU() # GELU

        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "optimized_simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1

        # 计算均值
        mu = x.mean(dim=[2, 3], keepdim=True)

        # 计算平方差
        x_minus_mu = x - mu
        x_minus_mu_square = x_minus_mu.pow(2)

        # 优化分母计算，避免除零错误
        denominator = 4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
        denominator = torch.clamp(denominator, min=1e-8)  # 避免除零

        # 计算注意力权重
        y = x_minus_mu_square / denominator + 0.5

        # 应用激活函数
        attention = self.activation(y)

        # 应用注意力权重到输入特征图
        return x * attention
# -----------------------------------OptSimAM start end --------------------------

# -----------------------------OptSimAM_ATT start---------------------------

class OptSimAM_ATT(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(OptSimAM_ATT, self).__init__()
        self.activation = nn.Sigmoid()  
        self.e_lambda = e_lambda
        # 添加通道注意力增强（针对无人机小目标）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 原有空间注意力计算
        b, c, h, w = x.size()
        n = w * h - 1

        # 计算均值
        mu = x.mean(dim=[2, 3], keepdim=True)

        # 计算平方差
        x_minus_mu = x - mu
        x_minus_mu_square = x_minus_mu.pow(2)

        # 优化分母计算，避免除零错误
        denominator = 4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
        denominator = torch.clamp(denominator, min=1e-8)  # 避免除零

        # 计算注意力权重
        y = x_minus_mu_square / denominator + 0.5

        spatial_att = self.activation(y)

        # 新增通道注意力融合
        channel_att = self.channel_att(x)

        # 双注意力融合
        return x * (spatial_att + channel_att) / 2
# -----------------------------OptSimAM_ATT end---------------------------

# -----------------------------HybridOptSimAM start---------------------------
class HybridOptSimAM(nn.Module):
    def __init__(self, channels=None):
        super().__init__()
        # 空间注意力分支
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels//16, 7, padding=3, groups=channels//16),
            nn.Mish(),
            nn.Conv2d(channels//16, 1, 1),
            nn.Sigmoid()
        )
        
        # 通道注意力分支
        self.channel_att = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, channels//8, 1),
            nn.Mish(),
            nn.Conv2d(channels//8, channels, 1),
            nn.Sigmoid()
        )
        
        # 动态融合参数（注册为可训练参数）
        self.fusion_gate = nn.Parameter(torch.tensor(0.5)) 

    def forward(self, x):
        # 空间注意力计算
        spatial_weight = self.spatial_att(x)
        
        # 通道注意力计算
        channel_weight = self.channel_att(x)
        
        # 动态融合
        fused_att = (spatial_weight * self.fusion_gate.sigmoid() + 
                    channel_weight * (1 - self.fusion_gate.sigmoid()))
        
        return x * fused_att + x
# -----------------------------Hybrid end---------------------------

# -----------------------------DynamicSimAM start---------------------------

class DynamicSimAM(nn.Module):
    def __init__(self, channels=None, e_lambda_init=1e-4):
        super(DynamicSimAM, self).__init__()
        # 将 e_lambda 设计为一个可学习的参数
        self.e_lambda = nn.Parameter(torch.tensor(e_lambda_init).float())
        self.activation = nn.Sigmoid()  # 可以根据需要替换为其他激活函数

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'e_lambda={self.e_lambda.item():.4e})'
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1

        # 计算均值和方差
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # 动态调整 e_lambda
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda) + 1e-6) + 0.5

        return x * self.activation(y)
# -----------------------------DynamicSimAM end---------------------------

# ----------------------------SimAM_EffectiveSE start-----------------------------
from timm.layers.create_act import create_act_layer

class EffectiveSEModule(nn.Module):
    def __init__(self, channels, add_maxpool=False, gate_layer='hard_sigmoid'):
        super(EffectiveSEModule, self).__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)
    
# 顺序串联 EffectiveSE->SimAM
class SimAM_EffectiveSE(nn.Module):
    def __init__(self, channels, e_lambda=1e-4, add_maxpool=False, gate_layer='hard_sigmoid'):
        super(SimAM_EffectiveSE, self).__init__()
        self.effective_se = EffectiveSEModule(channels, add_maxpool, gate_layer)
        self.simam = SimAM(channels, e_lambda)

    def forward(self, x):
        # 先通过EffectiveSE模块
        x = self.effective_se(x)
        # 再通过SimAM模块
        x = self.simam(x)
        return x    

# 顺序串联 SimAM->EffectiveSE
class SimAM_SE(nn.Module):
    def __init__(self, channels, e_lambda=1e-4, add_maxpool=False, gate_layer='hard_sigmoid'):
        super(SimAM_SE, self).__init__()
        self.effective_se = EffectiveSEModule(channels, add_maxpool, gate_layer)
        self.simam = SimAM(channels, e_lambda)

    def forward(self, x):
        # 先通过SimAM模块
        x = self.simam(x)
        # 再通过EffectiveSE模块
        x = self.effective_se(x)
        return x    


# 顺序串联 DynamicSimAM 和 EffectiveSE 模块
class DynamicSimAM_EffectiveSE(nn.Module):
    def __init__(self, channels, e_lambda_init=1e-4, add_maxpool=False, gate_layer='hard_sigmoid'):
        super(DynamicSimAM_EffectiveSE, self).__init__()
        self.effective_se = EffectiveSEModule(channels, add_maxpool, gate_layer)
        self.simam = DynamicSimAM(channels, e_lambda_init)

    def forward(self, x):
        # 先通过EffectiveSE模块
        x = self.effective_se(x)
        # 再通过SimAM模块
        x = self.simam(x)
        return x   
    
# 并行加权结合
class SimAM_EffectiveSE_Parallel(nn.Module):
    def __init__(self, channels, e_lambda=1e-4, add_maxpool=False, gate_layer='hard_sigmoid'):
        super(SimAM_EffectiveSE_Parallel, self).__init__()
        self.simam = SimAM(channels, e_lambda)
        self.effective_se = EffectiveSEModule(channels, add_maxpool, gate_layer)
        self.weight_simam = nn.Parameter(torch.ones(1))
        self.weight_effective_se = nn.Parameter(torch.ones(1))

    def forward(self, x):
        simam_out = self.simam(x)
        effective_se_out = self.effective_se(x)
        weighted_simam = self.weight_simam * simam_out
        weighted_effective_se = self.weight_effective_se * effective_se_out
        final_out = weighted_simam + weighted_effective_se
        return final_out
    
# 元素相乘结合
class SimAM_EffectiveSE_Mul(nn.Module):
    def __init__(self, channels, e_lambda=1e-4, add_maxpool=False, gate_layer='hard_sigmoid'):
        super(SimAM_EffectiveSE_Mul, self).__init__()
        self.simam = SimAM(channels, e_lambda)
        self.effective_se = EffectiveSEModule(channels, add_maxpool, gate_layer)

    def forward(self, x):
        simam_out = self.simam(x)
        effective_se_out = self.effective_se(x)
        final_out = simam_out * effective_se_out
        return final_out
# ----------------------------SimAM_EffectiveSE end-----------------------------

# -----------------------------SimAM_ELA start----------------------------------
# ELA 模块
class ELA(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(16, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.conv1x1(self.pool_h(x).reshape((b, c, h))).reshape((b, c, h, 1))
        x_w = self.conv1x1(self.pool_w(x).reshape((b, c, w))).reshape((b, c, 1, w))
        return x * x_h * x_w

# 顺序串联 SimAM 和 ELA 模块
class SimAM_ELA(nn.Module):
    def __init__(self, channels, e_lambda=1e-4):
        super(SimAM_ELA, self).__init__()
        self.simam = SimAM(channels, e_lambda)
        self.ela = ELA(channels)

    def forward(self, x):
        # 先通过 SimAM 模块
        x = self.simam(x)
        # 再通过 ELA 模块
        x = self.ela(x)
        # print(f'SimAM_ELA size: {x.size()}')
        return x

# 并行加权   
class WeightedSimAM_ELA(nn.Module):
    def __init__(self, channels, e_lambda=1e-4):
        super(WeightedSimAM_ELA, self).__init__()
        self.simam = SimAM(channels, e_lambda)
        self.ela = ELA(channels)
        # 可学习的权重
        self.weight_simam = nn.Parameter(torch.ones(1))
        self.weight_ela = nn.Parameter(torch.ones(1))

    def forward(self, x):
        simam_out = self.simam(x)
        ela_out = self.ela(x)
        weighted_simam = self.weight_simam * simam_out
        weighted_ela = self.weight_ela * ela_out
        merged_out = weighted_simam + weighted_ela
        # print(f'weighted_simam size: {weighted_simam.size()}')
        return merged_out

# 门控融合模块
class GatedSimAM_ELA(nn.Module):
    def __init__(self, channels, e_lambda=1e-4):
        super(GatedSimAM_ELA, self).__init__()
        self.simam = SimAM(channels, e_lambda)
        self.ela = ELA(channels)
        # 门控网络
        self.gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        simam_out = self.simam(x)
        ela_out = self.ela(x)
        gate_value = self.gate(x)
        # 根据门控值加权合并
        output = gate_value * simam_out + (1 - gate_value) * ela_out
        return output

# 特征交互模块
class FeatureInteractionSimAM_ELA(nn.Module):
    def __init__(self, channels, e_lambda=1e-4):
        super(FeatureInteractionSimAM_ELA, self).__init__()
        self.simam = SimAM(channels, e_lambda)
        self.ela = ELA(channels)
        # 用于特征交互的卷积层
        self.interaction_conv = nn.Conv2d(2 * channels, channels, kernel_size=1)

    def forward(self, x):
        simam_out = self.simam(x)
        ela_out = self.ela(x)
        # 拼接特征
        combined_features = torch.cat([simam_out, ela_out], dim=1)
        # 通过卷积层进行特征交互
        interacted_features = self.interaction_conv(combined_features)
        return interacted_features

# 残差融合   
class ResidualSimAM_ELA(nn.Module):
    def __init__(self, channels, e_lambda=1e-4):
        super(ResidualSimAM_ELA, self).__init__()
        self.simam = SimAM(channels, e_lambda)
        self.ela = ELA(channels)

    def forward(self, x):
        simam_out = self.simam(x)
        ela_out = self.ela(x)
        # 合并 SimAM 和 ELA 输出
        combined_out = simam_out + ela_out
        # 残差连接
        output = x + combined_out
        # print(f'ResidualSimAM_ELA size: {output.size()}')
        return output
# ----------------------------SimAM_ELA end-----------------------------------



# ----------------------------SimAM_ECA start-----------------------------------
# ECA 模块
class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
 
    def __init__(self, c1, c2, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# 顺序串联 
class SimAM_ECA(nn.Module):
    def __init__(self, channels, e_lambda=1e-4, k_size=3):
        super(SimAM_ECA, self).__init__()
        self.simam = SimAM(channels, e_lambda)
        self.eca = ECA(channels, channels, k_size)

    def forward(self, x):
        # 先通过 SimAM 模块
        simam_out = self.simam(x)
        # 再通过 ECA 模块
        eca_out = self.eca(simam_out)
        return eca_out
    
# 并行加权
class SimAM_ECA_ParallelWeighted(nn.Module):
    def __init__(self, channels, e_lambda=1e-4, k_size=3):
        super(SimAM_ECA_ParallelWeighted, self).__init__()
        self.simam = SimAM(channels, e_lambda)
        self.eca = ECA(channels, channels, k_size)
        # 可学习的权重
        self.weight_simam = nn.Parameter(torch.ones(1))
        self.weight_eca = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # 通过 SimAM 模块
        simam_out = self.simam(x)
        # 通过 ECA 模块
        eca_out = self.eca(x)
        # 加权
        weighted_simam = self.weight_simam * simam_out
        weighted_eca = self.weight_eca * eca_out
        # 相加
        output = weighted_simam + weighted_eca
        return output
    
# 自适应融合 SimAM 和 ECA 的模块
class SimAM_ECA_AdaptiveFusion(nn.Module):
    def __init__(self, channels, e_lambda=1e-4, k_size=3):
        super(SimAM_ECA_AdaptiveFusion, self).__init__()
        self.simam = SimAM(channels, e_lambda)
        self.eca = ECA(channels, channels, k_size)
        # 用于生成自适应权重的卷积层
        self.fusion_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 2, kernel_size=1),  # 输出2个通道，分别对应 SimAM 和 ECA 的权重
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 通过 SimAM 模块
        simam_out = self.simam(x)
        # 通过 ECA 模块
        eca_out = self.eca(x)
        # 生成自适应权重
        fusion_weights = self.fusion_conv(x)
        weight_simam, weight_eca = torch.split(fusion_weights, 1, dim=1)
        # 自适应融合
        output = weight_simam * simam_out + weight_eca * eca_out
        print(f'SimAM_ECA_AdaptiveFusion size: {output.size()}')
        return output
# -----------------------------SimAM_ECA end---------------------------

# ----------------------------SimAM_EMA start---------------------------
# EMA 模块
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

# 顺序串联 SimAM 和 EMA 模块
class EMA_SimAM(nn.Module):
    def __init__(self, channels, e_lambda=1e-4, factor=8):
        super(EMA_SimAM, self).__init__()
        self.simam = SimAM(channels, e_lambda)
        self.ema = EMA(channels, factor)

    def forward(self, x):
        ema_out = self.ema(x)
        simam_out = self.simam(ema_out)
        return simam_out

# 加权融合 SimAM 和 EMA 模块
class SimAM_EMA_Fusion(nn.Module):
    def __init__(self, channels, e_lambda=1e-4, factor=8):
        super(SimAM_EMA_Fusion, self).__init__()
        self.simam = SimAM(channels, e_lambda)
        self.ema = EMA(channels, factor)
        # 可学习的权重
        self.weight_simam = nn.Parameter(torch.ones(1))
        self.weight_ema = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # 通过 SimAM 模块
        simam_out = self.simam(x)
        # 通过 EMA 模块
        ema_out = self.ema(x)
        # 加权
        weighted_simam = self.weight_simam * simam_out
        weighted_ema = self.weight_ema * ema_out
        # 相加
        output = weighted_simam + weighted_ema
        return output

# 元素级乘积融合 SimAM 和 EMA 模块
class SimAM_EMA_ElementwiseMul(nn.Module):
    def __init__(self, channels, e_lambda=1e-4, factor=8):
        super(SimAM_EMA_ElementwiseMul, self).__init__()
        self.simam = SimAM(channels, e_lambda)
        self.ema = EMA(channels, factor)

    def forward(self, x):
        simam_out = self.simam(x)
        ema_out = self.ema(x)
        output = simam_out * ema_out
        return output    

# 自适应融合 SimAM 和 EMA 模块
class SimAM_EMA_AdaptiveFusion(nn.Module):
    def __init__(self, channels, e_lambda=1e-4, factor=8):
        super(SimAM_EMA_AdaptiveFusion, self).__init__()
        self.simam = SimAM(channels, e_lambda)
        self.ema = EMA(channels, factor)
        self.fusion_net = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        simam_out = self.simam(x)
        ema_out = self.ema(x)
        combined = torch.cat([simam_out, ema_out], dim=1)
        weights = self.fusion_net(combined)
        output = weights * simam_out + (1 - weights) * ema_out
        return output

# 门控融合 SimAM 和 EMA 模块
class SimAM_EMA_GatedFusion(nn.Module):
    def __init__(self, channels, e_lambda=1e-4, factor=8):
        super(SimAM_EMA_GatedFusion, self).__init__()
        self.simam = SimAM(channels, e_lambda)
        self.ema = EMA(channels, factor)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        simam_out = self.simam(x)
        ema_out = self.ema(x)
        gate_value = self.gate(x)
        output = gate_value * simam_out + (1 - gate_value) * ema_out
        return output
# -----------------------------SimAM_EMA end---------------------------
# -----------------------------SPPFCSPC start---------------------------
# class SPPFCSPC(nn.Module):是一种常用于目标检测和图像识别等任务的神经网络模块，
# 结合了空间金字塔池化快速（SPPF）和跨阶段部分连接（CSP）的思想，用于增强特征的表达能力。
class SPPFCSPC(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(SPPFCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1, x2, x3, self.m(x3)), 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


# -----------------------------SPPFCSPC end---------------------------

######################################## DySample start ########################################

class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            self.constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def normal_init(self, module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def constant_init(self, module, val, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing='ij')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='ij')
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").reshape((B, -1, self.scale * H, self.scale * W))

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

######################################## DySample end ########################################

# ----------------------------ASF start-----------------------------
def autopad_ASF(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv_ASF(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad_ASF(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class DownSample(nn.Module):
    def __init__(self, c1, c2, k=2):
        super(DownSample, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=k)
        self.avgpool = nn.AvgPool2d(kernel_size=k, stride=k)
        self.cv = Conv_ASF(c1, c2, 1, 1)

    def forward(self, x):
        x1 = self.maxpool(x)
        x2 = self.avgpool(x)
        x = x1 + x2
        x = self.cv(x)
        return x


# Zoom_cat:用于将不同尺度的特征图（大、中、小）整合到中间尺度上。
class Zoom_cat(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # self.conv_l_post_down = Conv(in_dim, 2*in_dim, 3, 1, 1)

    def forward(self, x):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        l, m, s = x[0], x[1], x[2]

        # print(f'l size: {m.size()}')

        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = F.interpolate(s, m.shape[2:], mode='nearest')

        lms = torch.cat([l, m, s], dim=1)

        # print(f'lms size: {lms.size()}')

        return lms


# ScalSeq:用于处理不同尺度的特征图（p3、p4、p5），将它们调整到相同大小后进行 3D 卷积处理。
class ScalSeq(nn.Module):
    def __init__(self, channel):
        super(ScalSeq, self).__init__()
        self.conv1 = Conv_ASF(56, channel, 1)
        self.conv2 = Conv_ASF(160, channel, 1)
        self.conv3 = Conv_ASF(24, channel, 1)
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))

    def forward(self, x):
        p3, p4, p5 = x[0], x[1], x[2]

        # print(f'p3 size: {p3.size()}')
        # print(f'p4 size: {p4.size()}')
        # print(f'p5 size: {p5.size()}')

        p3_2 = self.conv3(p3)

        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        # print(f'p4_2 size: {p4_2.size()}')

        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
        # print(f'p5_2 size: {p5_2.size()}')

        p3_3d = torch.unsqueeze(p3_2, -3)
        # print(f'p3_3d size: {p3_3d.size()}')

        p4_3d = torch.unsqueeze(p4_2, -3)
        # print(f'p4_3d size: {p4_3d.size()}')

        p5_3d = torch.unsqueeze(p5_2, -3)
        # print(f'p5_3d size: {p5_3d.size()}')

        combine = torch.cat([p3_3d, p4_3d, p5_3d], dim=2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        # print(f'x size: {x.size()}')
        return x


# Add:用于将两个输入特征图逐元素相加。
class Add(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, ch=256):
        super().__init__()

    def forward(self, x):
        input1, input2 = x[0], x[1]
        x = input1 + input2
        return x



# channel_att:用于对输入特征图进行通道注意力机制的处理。
class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# local_att:实现了局部注意力机制，用于增强特征图中重要区域的特征表达。。
class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


# attention_model:结合了通道注意力机制和局部注意力机制，对输入特征图进行处理。
class attention_model(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, ch=256):
        super().__init__()
        self.channel_att = channel_att(ch)
        self.local_att = local_att(ch)

    def forward(self, x):
        input1, input2 = x[0], x[1]
        input1 = self.channel_att(input1)
        x = input1 + input2
        x = self.local_att(x)
        return x

# ----------------------------ASF end-----------------------------
# ------------------------------------LSK start --------------------------
class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


# ------------------------------------LSK end --------------------------

# --------------------------PA start----------------------------------------------------------
class ParNetAttention(nn.Module):

    def __init__(self, channel=1024, out_channel=1024):
        super().__init__()
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, out_channel, kernel_size=1),
            nn.Sigmoid())
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel))
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel))
        self.silu = nn.SiLU()

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        y = self.silu(x1 + x2 + x3)
        return y

# 并行双注意力融合
class CombinedPa(nn.Module):
    def __init__(self, channel=1024, out_channel=1024):
        super().__init__()
        # 1x1卷积分支
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel)
        )
        # 3x3卷积分支
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel)
        )
        # 改进的SE分支（带通道调整）
        self.eff_se = nn.Sequential(
            EffectiveSEModule(channel),  # 原始EffectiveSE输出通道为channel
            nn.Conv2d(channel, out_channel, kernel_size=1)  # 调整到目标通道
        )
        self.silu = nn.SiLU()

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.eff_se(x)  # 输出通道已调整为out_channel
        return self.silu(x1 + x2 + x3)

# 级联式注意力增强
class CascadePa(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.se_pre = EffectiveSEModule(channel)  # 前置SE
        self.parnet = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.SiLU()
        )
        self.se_post = nn.Sequential(  # 轻量后置SE
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//4, 1),
            nn.ReLU(),
            nn.Conv2d(channel//4, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.se_pre(x)
        x = self.parnet(x)
        return x * self.se_post(x)

#  混合注意力残差结构 
class ResidualPa(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.parnet = ParNetAttention(channel, channel)
        self.eff_se = EffectiveSEModule(channel)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习融合系数

    def forward(self, x):
        par_out = self.parnet(x)
        se_out = self.eff_se(x)
        return x + self.alpha*par_out + (1-self.alpha)*se_out

# --------------------------PA end------------------------------------------------------------

# # -----------------------------GFPN start---------------------------------
# def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
#     '''Basic cell for rep-style block, including conv and bn'''
#     result = nn.Sequential()
#     result.add_module(
#         'conv',
#         nn.Conv2d(in_channels=in_channels,
#                   out_channels=out_channels,
#                   kernel_size=kernel_size,
#                   stride=stride,
#                   padding=padding,
#                   groups=groups,
#                   bias=False))
#     result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
#     return result


# class RepConv(nn.Module):
#     '''RepConv is a basic rep-style block, including training and deploy status
#     Code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
#     '''

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=3,
#                  stride=1,
#                  padding=1,
#                  dilation=1,
#                  groups=1,
#                  padding_mode='zeros',
#                  deploy=False,
#                  act='relu',
#                  norm=None):
#         super(RepConv, self).__init__()
#         self.deploy = deploy
#         self.groups = groups
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         assert kernel_size == 3
#         assert padding == 1

#         padding_11 = padding - kernel_size // 2

#         if isinstance(act, str):
#             self.nonlinearity = get_activation(act)
#         else:
#             self.nonlinearity = act

#         if deploy:
#             self.rbr_reparam = nn.Conv2d(in_channels=in_channels,
#                                          out_channels=out_channels,
#                                          kernel_size=kernel_size,
#                                          stride=stride,
#                                          padding=padding,
#                                          dilation=dilation,
#                                          groups=groups,
#                                          bias=True,
#                                          padding_mode=padding_mode)

#         else:
#             self.rbr_identity = None
#             self.rbr_dense = conv_bn(in_channels=in_channels,
#                                      out_channels=out_channels,
#                                      kernel_size=kernel_size,
#                                      stride=stride,
#                                      padding=padding,
#                                      groups=groups)
#             self.rbr_1x1 = conv_bn(in_channels=in_channels,
#                                    out_channels=out_channels,
#                                    kernel_size=1,
#                                    stride=stride,
#                                    padding=padding_11,
#                                    groups=groups)

#     def forward(self, inputs):
#         '''Forward process'''
#         if hasattr(self, 'rbr_reparam'):
#             return self.nonlinearity(self.rbr_reparam(inputs))

#         if self.rbr_identity is None:
#             id_out = 0
#         else:
#             id_out = self.rbr_identity(inputs)

#         return self.nonlinearity(
#             self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

#     def get_equivalent_kernel_bias(self):
#         kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
#         kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
#         kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
#         return kernel3x3 + self._pad_1x1_to_3x3_tensor(
#             kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

#     def _pad_1x1_to_3x3_tensor(self, kernel1x1):
#         if kernel1x1 is None:
#             return 0
#         else:
#             return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

#     def _fuse_bn_tensor(self, branch):
#         if branch is None:
#             return 0, 0
#         if isinstance(branch, nn.Sequential):
#             kernel = branch.conv.weight
#             running_mean = branch.bn.running_mean
#             running_var = branch.bn.running_var
#             gamma = branch.bn.weight
#             beta = branch.bn.bias
#             eps = branch.bn.eps
#         else:
#             assert isinstance(branch, nn.BatchNorm2d)
#             if not hasattr(self, 'id_tensor'):
#                 input_dim = self.in_channels // self.groups
#                 kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
#                                         dtype=np.float32)
#                 for i in range(self.in_channels):
#                     kernel_value[i, i % input_dim, 1, 1] = 1
#                 self.id_tensor = torch.from_numpy(kernel_value).to(
#                     branch.weight.device)
#             kernel = self.id_tensor
#             running_mean = branch.running_mean
#             running_var = branch.running_var
#             gamma = branch.weight
#             beta = branch.bias
#             eps = branch.eps
#         std = (running_var + eps).sqrt()
#         t = (gamma / std).reshape(-1, 1, 1, 1)
#         return kernel * t, beta - running_mean * gamma / std

#     def switch_to_deploy(self):
#         if hasattr(self, 'rbr_reparam'):
#             return
#         kernel, bias = self.get_equivalent_kernel_bias()
#         self.rbr_reparam = nn.Conv2d(
#             in_channels=self.rbr_dense.conv.in_channels,
#             out_channels=self.rbr_dense.conv.out_channels,
#             kernel_size=self.rbr_dense.conv.kernel_size,
#             stride=self.rbr_dense.conv.stride,
#             padding=self.rbr_dense.conv.padding,
#             dilation=self.rbr_dense.conv.dilation,
#             groups=self.rbr_dense.conv.groups,
#             bias=True)
#         self.rbr_reparam.weight.data = kernel
#         self.rbr_reparam.bias.data = bias
#         for para in self.parameters():
#             para.detach_()
#         self.__delattr__('rbr_dense')
#         self.__delattr__('rbr_1x1')
#         if hasattr(self, 'rbr_identity'):
#             self.__delattr__('rbr_identity')
#         if hasattr(self, 'id_tensor'):
#             self.__delattr__('id_tensor')
#         self.deploy = True


# class Swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(Swish, self).__init__()
#         self.inplace = inplace

#     def forward(self, x):
#         if self.inplace:
#             x.mul_(F.sigmoid(x))
#             return x
#         else:
#             return x * F.sigmoid(x)


# def get_activation(name='silu', inplace=True):
#     if name is None:
#         return nn.Identity()

#     if isinstance(name, str):
#         if name == 'silu':
#             module = nn.SiLU(inplace=inplace)
#         elif name == 'relu':
#             module = nn.ReLU(inplace=inplace)
#         elif name == 'lrelu':
#             module = nn.LeakyReLU(0.1, inplace=inplace)
#         elif name == 'swish':
#             module = Swish(inplace=inplace)
#         elif name == 'hardsigmoid':
#             module = nn.Hardsigmoid(inplace=inplace)
#         elif name == 'identity':
#             module = nn.Identity()
#         else:
#             raise AttributeError('Unsupported act type: {}'.format(name))
#         return module

#     elif isinstance(name, nn.Module):
#         return name

#     else:
#         raise AttributeError('Unsupported act type: {}'.format(name))


# def get_norm(name, out_channels, inplace=True):
#     if name == 'bn':
#         module = nn.BatchNorm2d(out_channels)
#     else:
#         raise NotImplementedError
#     return module


# class ConvBNAct(nn.Module):
#     """A Conv2d -> Batchnorm -> silu/leaky relu block"""

#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             ksize,
#             stride=1,
#             groups=1,
#             bias=False,
#             act='silu',
#             norm='bn',
#             reparam=False,
#     ):
#         super().__init__()
#         # same padding
#         pad = (ksize - 1) // 2
#         self.conv = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=ksize,
#             stride=stride,
#             padding=pad,
#             groups=groups,
#             bias=bias,
#         )
#         if norm is not None:
#             self.bn = get_norm(norm, out_channels, inplace=True)
#         if act is not None:
#             self.act = get_activation(act, inplace=True)
#         self.with_norm = norm is not None
#         self.with_act = act is not None

#     def forward(self, x):
#         x = self.conv(x)
#         if self.with_norm:
#             x = self.bn(x)
#         if self.with_act:
#             x = self.act(x)
#         return x

#     def fuseforward(self, x):
#         return self.act(self.conv(x))


# class BasicBlock_3x3_Reverse(nn.Module):
#     def __init__(self,
#                  ch_in,
#                  ch_hidden_ratio,
#                  ch_out,
#                  act='relu',
#                  shortcut=True):
#         super(BasicBlock_3x3_Reverse, self).__init__()
#         assert ch_in == ch_out
#         ch_hidden = int(ch_in * ch_hidden_ratio)
#         self.conv1 = ConvBNAct(ch_hidden, ch_out, 3, stride=1, act=act)
#         self.conv2 = RepConv(ch_in, ch_hidden, 3, stride=1, act=act)
#         self.shortcut = shortcut

#     def forward(self, x):
#         y = self.conv2(x)
#         y = self.conv1(y)
#         if self.shortcut:
#             return x + y
#         else:
#             return y


# class SPP(nn.Module):
#     def __init__(
#             self,
#             ch_in,
#             ch_out,
#             k,
#             pool_size,
#             act='swish',
#     ):
#         super(SPP, self).__init__()
#         self.pool = []
#         for i, size in enumerate(pool_size):
#             pool = nn.MaxPool2d(kernel_size=size,
#                                 stride=1,
#                                 padding=size // 2,
#                                 ceil_mode=False)
#             self.add_module('pool{}'.format(i), pool)
#             self.pool.append(pool)
#         self.conv = ConvBNAct(ch_in, ch_out, k, act=act)

#     def forward(self, x):
#         outs = [x]

#         for pool in self.pool:
#             outs.append(pool(x))
#         y = torch.cat(outs, axis=1)

#         y = self.conv(y)
#         return y


# class CSPStage(nn.Module):
#     def __init__(self,
#                  ch_in,
#                  ch_out,
#                  n=1,
#                  block_fn='BasicBlock_3x3_Reverse',
#                  ch_hidden_ratio=1.0,
#                  act='silu',
#                  spp=False):
#         super(CSPStage, self).__init__()

#         split_ratio = 2
#         ch_first = int(ch_out // split_ratio)
#         ch_mid = int(ch_out - ch_first)
#         self.conv1 = ConvBNAct(ch_in, ch_first, 1, act=act)
#         self.conv2 = ConvBNAct(ch_in, ch_mid, 1, act=act)
#         self.convs = nn.Sequential()

#         next_ch_in = ch_mid
#         for i in range(n):
#             if block_fn == 'BasicBlock_3x3_Reverse':
#                 self.convs.add_module(
#                     str(i),
#                     BasicBlock_3x3_Reverse(next_ch_in,
#                                            ch_hidden_ratio,
#                                            ch_mid,
#                                            act=act,
#                                            shortcut=True))
#             else:
#                 raise NotImplementedError
#             if i == (n - 1) // 2 and spp:
#                 self.convs.add_module(
#                     'spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
#             next_ch_in = ch_mid
#         self.conv3 = ConvBNAct(ch_mid * n + ch_first, ch_out, 1, act=act)

#     def forward(self, x):
#         y1 = self.conv1(x)
#         y2 = self.conv2(x)

#         mid_out = [y1]
#         for conv in self.convs:
#             y2 = conv(y2)
#             mid_out.append(y2)
#         y = torch.cat(mid_out, axis=1)
#         y = self.conv3(y)
#         return y

# # -----------------------------GFPN end---------------------------------

# --------------------------------ContextAggregation start----------------------------------------------
from mmcv.cnn import ConvModule
from mmengine.model import caffe2_xavier_init, constant_init


class ContextAggregation(nn.Module):
    """
    Context Aggregation Block.

    Args:
        in_channels (int): Number of input channels.
        reduction (int, optional): Channel reduction ratio. Default: 1.
        conv_cfg (dict or None, optional): Config dict for the convolution
            layer. Default: None.
    """

    def __init__(self, in_channels, reduction=1):
        super(ContextAggregation, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)

        conv_params = dict(kernel_size=1, act_cfg=None)

        self.a = ConvModule(in_channels, 1, **conv_params)
        self.k = ConvModule(in_channels, 1, **conv_params)
        self.v = ConvModule(in_channels, self.inter_channels, **conv_params)
        self.m = ConvModule(self.inter_channels, in_channels, **conv_params)

        self.init_weights()

    def init_weights(self):
        for m in (self.a, self.k, self.v):
            caffe2_xavier_init(m.conv)
        constant_init(self.m.conv, 0)

    def forward(self, x):
        n, c = x.size(0), self.inter_channels

        # a: [N, 1, H, W]
        a = self.a(x).sigmoid()

        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)

        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)

        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)
        y = self.m(y) * a

        return x + y
# --------------------------------ContextAggregation end----------------------------------------------

#--------------------------------AIFI start---------------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]
#--------------------------------AIFI end---------------------------------------------------------

# -------------------------------CAM start--------------------------------------------------------------
class CAM(nn.Module):
    def __init__(self, inc, fusion='weight'):
        super().__init__()

        assert fusion in ['weight', 'adaptive', 'concat']
        self.fusion = fusion

        self.conv1 = Conv(inc, inc, 3, 1, None, 1, 1)
        self.conv2 = Conv(inc, inc, 3, 1, None, 1, 3)
        self.conv3 = Conv(inc, inc, 3, 1, None, 1, 5)

        self.fusion_1 = Conv(inc, inc, 1)
        self.fusion_2 = Conv(inc, inc, 1)
        self.fusion_3 = Conv(inc, inc, 1)

        if self.fusion == 'adaptive':
            self.fusion_4 = Conv(inc * 3, 3, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        if self.fusion == 'weight':
            return self.fusion_1(x1) + self.fusion_2(x2) + self.fusion_3(x3)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(
                self.fusion_4(torch.cat([self.fusion_1(x1), self.fusion_2(x2), self.fusion_3(x3)], dim=1)), dim=1)
            x1_weight, x2_weight, x3_weight = torch.split(fusion, [1, 1, 1], dim=1)
            return x1 * x1_weight + x2 * x2_weight + x3 * x3_weight
        else:
            return torch.cat([self.fusion_1(x1), self.fusion_2(x2), self.fusion_3(x3)], dim=1)
# ---------------------------------CAM end--------------------------------------------------------------

# # -------------------------------Goldyolo start---------------------------------------------------------
# def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
#     '''Basic cell for rep-style block, including conv and bn'''
#     result = nn.Sequential()
#     result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                                         kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
#                                         bias=bias))
#     result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
#     return result


# class RepVGGBlock(nn.Module):
#     '''RepVGGBlock is a basic rep-style block, including training and deploy status
#     This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
#     '''

#     def __init__(self, in_channels, out_channels, kernel_size=3,
#                  stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
#         super(RepVGGBlock, self).__init__()
#         """ Initialization of the class.
#         Args:
#             in_channels (int): Number of channels in the input image
#             out_channels (int): Number of channels produced by the convolution
#             kernel_size (int or tuple): Size of the convolving kernel
#             stride (int or tuple, optional): Stride of the convolution. Default: 1
#             padding (int or tuple, optional): Zero-padding added to both sides of
#                 the input. Default: 1
#             dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
#             groups (int, optional): Number of blocked connections from input
#                 channels to output channels. Default: 1
#             padding_mode (string, optional): Default: 'zeros'
#             deploy: Whether to be deploy status or training status. Default: False
#             use_se: Whether to use se. Default: False
#         """
#         self.deploy = deploy
#         self.groups = groups
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         assert kernel_size == 3
#         assert padding == 1

#         padding_11 = padding - kernel_size // 2

#         self.nonlinearity = nn.ReLU()

#         if use_se:
#             raise NotImplementedError("se block not supported yet")
#         else:
#             self.se = nn.Identity()

#         if deploy:
#             self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
#                                          stride=stride,
#                                          padding=padding, dilation=dilation, groups=groups, bias=True,
#                                          padding_mode=padding_mode)

#         else:
#             self.rbr_identity = nn.BatchNorm2d(
#                 num_features=in_channels) if out_channels == in_channels and stride == 1 else None
#             self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
#                                      stride=stride, padding=padding, groups=groups)
#             self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
#                                    padding=padding_11, groups=groups)

#     def forward(self, inputs):
#         '''Forward process'''
#         if hasattr(self, 'rbr_reparam'):
#             return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

#         if self.rbr_identity is None:
#             id_out = 0
#         else:
#             id_out = self.rbr_identity(inputs)

#         return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

#     def get_equivalent_kernel_bias(self):
#         kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
#         kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
#         kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
#         return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

#     def _pad_1x1_to_3x3_tensor(self, kernel1x1):
#         if kernel1x1 is None:
#             return 0
#         else:
#             return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

#     def _fuse_bn_tensor(self, branch):
#         if branch is None:
#             return 0, 0
#         if isinstance(branch, nn.Sequential):
#             kernel = branch.conv.weight
#             running_mean = branch.bn.running_mean
#             running_var = branch.bn.running_var
#             gamma = branch.bn.weight
#             beta = branch.bn.bias
#             eps = branch.bn.eps
#         else:
#             assert isinstance(branch, nn.BatchNorm2d)
#             if not hasattr(self, 'id_tensor'):
#                 input_dim = self.in_channels // self.groups
#                 kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
#                 for i in range(self.in_channels):
#                     kernel_value[i, i % input_dim, 1, 1] = 1
#                 self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
#             kernel = self.id_tensor
#             running_mean = branch.running_mean
#             running_var = branch.running_var
#             gamma = branch.weight
#             beta = branch.bias
#             eps = branch.eps
#         std = (running_var + eps).sqrt()
#         t = (gamma / std).reshape(-1, 1, 1, 1)
#         return kernel * t, beta - running_mean * gamma / std

#     def switch_to_deploy(self):
#         if hasattr(self, 'rbr_reparam'):
#             return
#         kernel, bias = self.get_equivalent_kernel_bias()
#         self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
#                                      out_channels=self.rbr_dense.conv.out_channels,
#                                      kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
#                                      padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
#                                      groups=self.rbr_dense.conv.groups, bias=True)
#         self.rbr_reparam.weight.data = kernel
#         self.rbr_reparam.bias.data = bias
#         for para in self.parameters():
#             para.detach_()
#         self.__delattr__('rbr_dense')
#         self.__delattr__('rbr_1x1')
#         if hasattr(self, 'rbr_identity'):
#             self.__delattr__('rbr_identity')
#         if hasattr(self, 'id_tensor'):
#             self.__delattr__('id_tensor')
#         self.deploy = True


# def onnx_AdaptiveAvgPool2d(x, output_size):
#     stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
#     kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
#     avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
#     x = avg(x)
#     return x


# def get_avg_pool():
#     if torch.onnx.is_in_onnx_export():
#         avg_pool = onnx_AdaptiveAvgPool2d
#     else:
#         avg_pool = nn.functional.adaptive_avg_pool2d
#     return avg_pool


# class SimFusion_3in(nn.Module):
#     def __init__(self, in_channel_list, out_channels):
#         super().__init__()
#         self.cv1 = Conv(in_channel_list[0], out_channels, act=nn.ReLU()) if in_channel_list[
#                                                                                 0] != out_channels else nn.Identity()
#         self.cv2 = Conv(in_channel_list[1], out_channels, act=nn.ReLU()) if in_channel_list[
#                                                                                 1] != out_channels else nn.Identity()
#         self.cv3 = Conv(in_channel_list[2], out_channels, act=nn.ReLU()) if in_channel_list[
#                                                                                 2] != out_channels else nn.Identity()
#         self.cv_fuse = Conv(out_channels * 3, out_channels, act=nn.ReLU())
#         self.downsample = nn.functional.adaptive_avg_pool2d

#     def forward(self, x):
#         N, C, H, W = x[1].shape
#         output_size = (H, W)

#         if torch.onnx.is_in_onnx_export():
#             self.downsample = onnx_AdaptiveAvgPool2d
#             output_size = np.array([H, W])

#         x0 = self.cv1(self.downsample(x[0], output_size))
#         x1 = self.cv2(x[1])
#         x2 = self.cv3(F.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False))
#         return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))


# class SimFusion_4in(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.avg_pool = nn.functional.adaptive_avg_pool2d

#     def forward(self, x):
#         x_l, x_m, x_s, x_n = x
#         B, C, H, W = x_s.shape
#         output_size = np.array([H, W])

#         if torch.onnx.is_in_onnx_export():
#             self.avg_pool = onnx_AdaptiveAvgPool2d

#         x_l = self.avg_pool(x_l, output_size)
#         x_m = self.avg_pool(x_m, output_size)
#         x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)

#         out = torch.cat([x_l, x_m, x_s, x_n], 1)
#         return out


# class IFM(nn.Module):
#     def __init__(self, inc, ouc, embed_dim_p=96, fuse_block_num=3) -> None:
#         super().__init__()

#         self.conv = nn.Sequential(
#             Conv(inc, embed_dim_p),
#             *[RepVGGBlock(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],
#             Conv(embed_dim_p, sum(ouc))
#         )

#     def forward(self, x):
#         return self.conv(x)


# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)

#     def forward(self, x):
#         return self.relu(x + 3) / 6


# class InjectionMultiSum_Auto_pool(nn.Module):
#     def __init__(
#             self,
#             inp: int,
#             oup: int,
#             global_inp: list,
#             flag: int
#     ) -> None:
#         super().__init__()
#         self.global_inp = global_inp
#         self.flag = flag
#         self.local_embedding = Conv(inp, oup, 1, act=False)
#         self.global_embedding = Conv(global_inp[self.flag], oup, 1, act=False)
#         self.global_act = Conv(global_inp[self.flag], oup, 1, act=False)
#         self.act = h_sigmoid()

#     def forward(self, x):
#         '''
#         x_g: global features
#         x_l: local features
#         '''
#         x_l, x_g = x
#         B, C, H, W = x_l.shape
#         g_B, g_C, g_H, g_W = x_g.shape
#         use_pool = H < g_H

#         gloabl_info = x_g.split(self.global_inp, dim=1)[self.flag]

#         local_feat = self.local_embedding(x_l)

#         global_act = self.global_act(gloabl_info)
#         global_feat = self.global_embedding(gloabl_info)

#         if use_pool:
#             avg_pool = get_avg_pool()
#             output_size = np.array([H, W])

#             sig_act = avg_pool(global_act, output_size)
#             global_feat = avg_pool(global_feat, output_size)

#         else:
#             sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
#             global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

#         out = local_feat * sig_act + global_feat
#         return out


# def get_shape(tensor):
#     shape = tensor.shape
#     if torch.onnx.is_in_onnx_export():
#         shape = [i.cpu().numpy() for i in shape]
#     return shape


# class PyramidPoolAgg(nn.Module):
#     def __init__(self, inc, ouc, stride, pool_mode='torch'):
#         super().__init__()
#         self.stride = stride
#         if pool_mode == 'torch':
#             self.pool = nn.functional.adaptive_avg_pool2d
#         elif pool_mode == 'onnx':
#             self.pool = onnx_AdaptiveAvgPool2d
#         self.conv = Conv(inc, ouc)

#     def forward(self, inputs):
#         B, C, H, W = get_shape(inputs[-1])
#         H = (H - 1) // self.stride + 1
#         W = (W - 1) // self.stride + 1

#         output_size = np.array([H, W])

#         if not hasattr(self, 'pool'):
#             self.pool = nn.functional.adaptive_avg_pool2d

#         if torch.onnx.is_in_onnx_export():
#             self.pool = onnx_AdaptiveAvgPool2d

#         out = [self.pool(inp, output_size) for inp in inputs]

#         return self.conv(torch.cat(out, dim=1))


# def drop_path(x, drop_prob: float = 0., training: bool = False):
#     """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output


# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = Conv(in_features, hidden_features, act=False)
#         self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
#         self.act = nn.ReLU6()
#         self.fc2 = Conv(hidden_features, out_features, act=False)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.dwconv(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


# class DropPath(nn.Module):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """

#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob

#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)


# class Attention(torch.nn.Module):
#     def __init__(self, dim, key_dim, num_heads, attn_ratio=4):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = key_dim ** -0.5
#         self.key_dim = key_dim
#         self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
#         self.d = int(attn_ratio * key_dim)
#         self.dh = int(attn_ratio * key_dim) * num_heads
#         self.attn_ratio = attn_ratio

#         self.to_q = Conv(dim, nh_kd, 1, act=False)
#         self.to_k = Conv(dim, nh_kd, 1, act=False)
#         self.to_v = Conv(dim, self.dh, 1, act=False)

#         self.proj = torch.nn.Sequential(nn.ReLU6(), Conv(self.dh, dim, act=False))

#     def forward(self, x):  # x (B,N,C)
#         B, C, H, W = get_shape(x)

#         qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
#         kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
#         vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

#         attn = torch.matmul(qq, kk)
#         attn = attn.softmax(dim=-1)  # dim = k

#         xx = torch.matmul(attn, vv)

#         xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
#         xx = self.proj(xx)
#         return xx


# class top_Block(nn.Module):

#     def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
#                  drop_path=0.):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.mlp_ratio = mlp_ratio

#         self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio)

#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

#     def forward(self, x1):
#         x1 = x1 + self.drop_path(self.attn(x1))
#         x1 = x1 + self.drop_path(self.mlp(x1))
#         return x1


# class TopBasicLayer(nn.Module):
#     def __init__(self, embedding_dim, ouc_list, block_num=2, key_dim=8, num_heads=4,
#                  mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.):
#         super().__init__()
#         self.block_num = block_num

#         self.transformer_blocks = nn.ModuleList()
#         for i in range(self.block_num):
#             self.transformer_blocks.append(top_Block(
#                 embedding_dim, key_dim=key_dim, num_heads=num_heads,
#                 mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
#                 drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path))
#         self.conv = nn.Conv2d(embedding_dim, sum(ouc_list), 1)

#     def forward(self, x):
#         # token * N
#         for i in range(self.block_num):
#             x = self.transformer_blocks[i](x)
#         return self.conv(x)


# class AdvPoolFusion(nn.Module):
#     def forward(self, x):
#         x1, x2 = x
#         if torch.onnx.is_in_onnx_export():
#             self.pool = onnx_AdaptiveAvgPool2d
#         else:
#             self.pool = nn.functional.adaptive_avg_pool2d

#         N, C, H, W = x2.shape
#         output_size = np.array([H, W])
#         x1 = self.pool(x1, output_size)

#         return torch.cat([x1, x2], 1)
# # -------------------------------Goldyolo end---------------------------------------------------------

#-------------------------------------RepVGGBlock start-------------------------------------------------
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))

    return result


class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True
# -------------------------------------RepVGGBlock end-------------------------------------

# ----------------------------Slim start-----------------------------
# 注意类名, 函数重复如果不用先注释：def autopad、class Conv、class DWConv、class GSConv、class GSConvns、 class GSBottleneck、
# def autopad(k, p=None, d=1):
#     """
#     Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.
#     `k`: kernel, `p`: padding, `d`: dilation.
#     """
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p


# class Conv(nn.Module):
#     # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
#     default_act = nn.SiLU()  # default activation

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         """Initializes a standard convolution layer with optional batch normalization and activation."""
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         """Applies a fused convolution and activation function to the input tensor `x`."""
#         return self.act(self.conv(x))


# class DWConv(Conv):
#     """Depth-wise convolution."""

#     def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
#         """Initialize Depth-wise convolution with given parameters."""
#         super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


# class GSConv(nn.Module):
#     # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         c_ = c2 // 2
#         self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
#         self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

#     def forward(self, x):
#         x1 = self.cv1(x)
#         x2 = torch.cat((x1, self.cv2(x1)), 1)
#         # shuffle
#         # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
#         # y = y.permute(0, 2, 1, 3, 4)
#         # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

#         b, n, h, w = x2.size()
#         b_n = b * n // 2
#         y = x2.reshape(b_n, 2, h * w)
#         y = y.permute(1, 0, 2)
#         y = y.reshape(2, -1, n // 2, h, w)

#         return torch.cat((y[0], y[1]), 1)


# class GSConvns(GSConv):
#     # GSConv with a normative-shuffle https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
#         super().__init__(c1, c2, k, s, p, g, act=True)
#         c_ = c2 // 2
#         self.shuf = nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)

#     def forward(self, x):
#         x1 = self.cv1(x)
#         x2 = torch.cat((x1, self.cv2(x1)), 1)
#         # normative-shuffle, TRT supported
#         return nn.ReLU()(self.shuf(x2))


# class GSBottleneck(nn.Module):
#     # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=3, s=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)
#         # for lighting
#         self.conv_lighting = nn.Sequential(
#             GSConv(c1, c_, 1, 1),
#             GSConv(c_, c2, 3, 1, act=False))
#         self.shortcut = Conv(c1, c2, 1, 1, act=False)

#     def forward(self, x):
#         return self.conv_lighting(x) + self.shortcut(x)


# class GSBottleneckns(GSBottleneck):
#     # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=3, s=1, e=0.5):
#         super().__init__(c1, c2, k, s, e)
#         c_ = int(c2 * e)
#         # for lighting
#         self.conv_lighting = nn.Sequential(
#             GSConvns(c1, c_, 1, 1),
#             GSConvns(c_, c2, 3, 1, act=False))


# class GSBottleneckC(GSBottleneck):
#     # cheap GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=3, s=1):
#         super().__init__(c1, c2, k, s)
#         self.shortcut = DWConv(c1, c2, k, s, act=False)


# class VoVGSCSP(nn.Module):
#     # VoVGSCSP module with GSBottleneck
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
#         self.res = Conv(c_, c_, 3, 1, act=False)
#         self.cv3 = Conv(2 * c_, c2, 1)

#     def forward(self, x):
#         x1 = self.gsb(self.cv1(x))
#         y = self.cv2(x)
#         return self.cv3(torch.cat((y, x1), dim=1))


# class VoVGSCSPns(VoVGSCSP):
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.gsb = nn.Sequential(*(GSBottleneckns(c_, c_, e=1.0) for _ in range(n)))


# class VoVGSCSPC(VoVGSCSP):
#     # cheap VoVGSCSP module with GSBottleneck
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__(c1, c2)
#         c_ = int(c2 * 0.5)  # hidden channels
#         self.gsb = GSBottleneckC(c_, c_, 1, 1)
# -------------------------------Slim end--------------------------------------------------

######################################## leveraging Visual Mamba Blocks start ########################################
# from mamba_vss import VSSBlock

# class C2f(nn.Module):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""

#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
#         expansion.
#         """
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

#     def forward(self, x):
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))

#     def forward_split(self, x):
#         """Forward pass using split() instead of chunk()."""
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))

# class Bottleneck_VSS(Bottleneck):
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         super().__init__(c1, c2, shortcut, g, k, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.cv2 = VSSBlock(c2)

# class C3_VSS(C3):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.m = nn.Sequential(*(Bottleneck_VSS(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

# class C2f_VSS(C2f):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(Bottleneck_VSS(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

# class C3_LVMB(C3):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.m = nn.Sequential(*(VSSBlock(c_) for _ in range(n)))

# class C2f_LVMB(C2f):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(VSSBlock(self.c) for _ in range(n))

######################################## leveraging Visual Mamba Blocks end ########################################

######################################## leveraging Visual Mamba Blocks start ########################################

# class Bottleneck_VSS(Bottleneck):
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         super().__init__(c1, c2, shortcut, g, k, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.cv2 = VSSBlock(c2)

# class C3_VSS(C3):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.m = nn.Sequential(*(Bottleneck_VSS(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

# class C2f_VSS(C2f):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(Bottleneck_VSS(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

# class C3_LVMB(C3):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.m = nn.Sequential(*(VSSBlock(c_) for _ in range(n)))

# class C2f_LVMB(C2f):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(VSSBlock(self.c) for _ in range(n))

######################################## leveraging Visual Mamba Blocks end ########################################

######################################## RCSOSA start ########################################
from torch.nn import init

class RepConv_SOSA(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")

class SEAttention(nn.Module):
    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SR(nn.Module):
    # Shuffle RepVGG
    def __init__(self, c1, c2):
        super().__init__()
        c1_ = int(c1 // 2)
        c2_ = int(c2 // 2)
        self.repconv = RepConv_SOSA(c1_, c2_, bn=True)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((x1, self.repconv(x2)), dim=1)
        out = self.channel_shuffle(out, 2)
        return out

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

class RCSOSA(nn.Module):
    # VoVNet with Res Shuffle RepVGG
    def __init__(self, c1, c2, n=1, se=False, g=1, e=0.5):
        super().__init__()
        n_ = n // 2
        c_ = make_divisible(int(c1 * e), 8)
        self.conv1 = RepConv_SOSA(c1, c_, bn=True)
        self.conv3 = RepConv_SOSA(int(c_ * 3), c2, bn=True)
        self.sr1 = nn.Sequential(*[SR(c_, c_) for _ in range(n_)])
        self.sr2 = nn.Sequential(*[SR(c_, c_) for _ in range(n_)])

        self.se = None
        if se:
            self.se = SEAttention(c2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.sr1(x1)
        x3 = self.sr2(x2)
        x = torch.cat((x1, x2, x3), 1)
        return self.conv3(x) if self.se is None else self.se(self.conv3(x))
    
######################################## RCSOSA end ###############################


######################################## SPPF with LSKA start ########################################
class LSKA(nn.Module):
    # Large-Separable-Kernel-Attention
    # https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention/tree/main
    def __init__(self, dim, k_size=7):
        super().__init__()

        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,2), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=(2,0), groups=dim, dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,4), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=(4,0), groups=dim, dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1,1), padding=(0,9), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1,1), padding=(9,0), groups=dim, dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1,1), padding=(0,15), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1,1), padding=(15,0), groups=dim, dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1,1), padding=(0,18), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1,1), padding=(18,0), groups=dim, dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1,1), padding=(0,24), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1,1), padding=(24,0), groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn

class SPPF_LSKA(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lska = LSKA(c_ * 4, k_size=11)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(self.lska(torch.cat((x, y1, y2, self.m(y2)), 1)))

######################################## SPPF with LSKA end ########################################

######################################## Semantics and Detail Infusion start ########################################
# 搭配Slim 使用

# class SDI(nn.Module):
#     def __init__(self, channels):
#         super().__init__()

#         # self.convs = nn.ModuleList([nn.Conv2d(channel, channels[0], kernel_size=3, stride=1, padding=1) for channel in channels])
#         self.convs = nn.ModuleList([GSConv(channel, channels[0]) for channel in channels])

#     def forward(self, xs):
#         ans = torch.ones_like(xs[0])
#         target_size = xs[0].shape[2:]
#         for i, x in enumerate(xs):
#             if x.shape[-1] > target_size[-1]:
#                 x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
#             elif x.shape[-1] < target_size[-1]:
#                 x = F.interpolate(x, size=(target_size[0], target_size[1]),
#                                       mode='bilinear', align_corners=True)
#             ans = ans * self.convs[i](x)
#         return ans

######################################## Semantics and Detail Infusion end ########################################

# --------------------------- C3_OREPA start----------------------
def myautopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std


class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def transVI_multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(kernel, [W_pixels_to_pad, W_pixels_to_pad, H_pixels_to_pad, H_pixels_to_pad])


class OREPA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=None,
                 groups=1,
                 dilation=1,
                 act=True,
                 internal_channels_1x1_3x3=None,
                 deploy=False,
                 single_init=False,
                 weight_only=False,
                 init_hyper_para=1.0, init_hyper_gamma=1.0):
        super(OREPA, self).__init__()
        self.deploy = deploy

        self.nonlinear = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.weight_only = weight_only

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.stride = stride
        padding = myautopad(kernel_size, padding, dilation)
        self.padding = padding
        self.dilation = dilation

        if deploy:
            self.orepa_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.branch_counter = 0

            self.weight_orepa_origin = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), kernel_size, kernel_size))
            init.kaiming_uniform_(self.weight_orepa_origin, a=math.sqrt(0.0))
            self.branch_counter += 1

            self.weight_orepa_avg_conv = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1,
                             1))
            self.weight_orepa_pfir_conv = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1,
                             1))
            init.kaiming_uniform_(self.weight_orepa_avg_conv, a=0.0)
            init.kaiming_uniform_(self.weight_orepa_pfir_conv, a=0.0)
            self.register_buffer(
                'weight_orepa_avg_avg',
                torch.ones(kernel_size,
                           kernel_size).mul(1.0 / kernel_size / kernel_size))
            self.branch_counter += 1
            self.branch_counter += 1

            self.weight_orepa_1x1 = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1,
                             1))
            init.kaiming_uniform_(self.weight_orepa_1x1, a=0.0)
            self.branch_counter += 1

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups <= 4 else 2 * in_channels

            if internal_channels_1x1_3x3 == in_channels:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(
                    torch.zeros(in_channels, int(in_channels / self.groups), 1, 1))
                id_value = np.zeros(
                    (in_channels, int(in_channels / self.groups), 1, 1))
                for i in range(in_channels):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(
                    self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer('id_tensor', id_tensor)

            else:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(
                    torch.zeros(internal_channels_1x1_3x3,
                                int(in_channels / self.groups), 1, 1))
                id_value = np.zeros(
                    (internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1))
                for i in range(internal_channels_1x1_3x3):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(
                    self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer('id_tensor', id_tensor)
                # init.kaiming_uniform_(
                # self.weight_orepa_1x1_kxk_conv1, a=math.sqrt(0.0))
            self.weight_orepa_1x1_kxk_conv2 = nn.Parameter(
                torch.Tensor(out_channels,
                             int(internal_channels_1x1_3x3 / self.groups),
                             kernel_size, kernel_size))
            init.kaiming_uniform_(self.weight_orepa_1x1_kxk_conv2, a=math.sqrt(0.0))
            self.branch_counter += 1

            expand_ratio = 8
            self.weight_orepa_gconv_dw = nn.Parameter(
                torch.Tensor(in_channels * expand_ratio, 1, kernel_size,
                             kernel_size))
            self.weight_orepa_gconv_pw = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels * expand_ratio / self.groups), 1, 1))
            init.kaiming_uniform_(self.weight_orepa_gconv_dw, a=math.sqrt(0.0))
            init.kaiming_uniform_(self.weight_orepa_gconv_pw, a=math.sqrt(0.0))
            self.branch_counter += 1

            self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
            if weight_only is False:
                self.bn = nn.BatchNorm2d(self.out_channels)

            self.fre_init()

            init.constant_(self.vector[0, :], 0.25 * math.sqrt(init_hyper_gamma))  # origin
            init.constant_(self.vector[1, :], 0.25 * math.sqrt(init_hyper_gamma))  # avg
            init.constant_(self.vector[2, :], 0.0 * math.sqrt(init_hyper_gamma))  # prior
            init.constant_(self.vector[3, :], 0.5 * math.sqrt(init_hyper_gamma))  # 1x1_kxk
            init.constant_(self.vector[4, :], 1.0 * math.sqrt(init_hyper_gamma))  # 1x1
            init.constant_(self.vector[5, :], 0.5 * math.sqrt(init_hyper_gamma))  # dws_conv

            self.weight_orepa_1x1.data = self.weight_orepa_1x1.mul(init_hyper_para)
            self.weight_orepa_origin.data = self.weight_orepa_origin.mul(init_hyper_para)
            self.weight_orepa_1x1_kxk_conv2.data = self.weight_orepa_1x1_kxk_conv2.mul(init_hyper_para)
            self.weight_orepa_avg_conv.data = self.weight_orepa_avg_conv.mul(init_hyper_para)
            self.weight_orepa_pfir_conv.data = self.weight_orepa_pfir_conv.mul(init_hyper_para)

            self.weight_orepa_gconv_dw.data = self.weight_orepa_gconv_dw.mul(math.sqrt(init_hyper_para))
            self.weight_orepa_gconv_pw.data = self.weight_orepa_gconv_pw.mul(math.sqrt(init_hyper_para))

            if single_init:
                #   Initialize the vector.weight of origin as 1 and others as 0. This is not the default setting.
                self.single_init()

    def fre_init(self):
        prior_tensor = torch.Tensor(self.out_channels, self.kernel_size,
                                    self.kernel_size)
        half_fg = self.out_channels / 2
        for i in range(self.out_channels):
            for h in range(3):
                for w in range(3):
                    if i < half_fg:
                        prior_tensor[i, h, w] = math.cos(math.pi * (h + 0.5) *
                                                         (i + 1) / 3)
                    else:
                        prior_tensor[i, h, w] = math.cos(math.pi * (w + 0.5) *
                                                         (i + 1 - half_fg) / 3)

        self.register_buffer('weight_orepa_prior', prior_tensor)

    def weight_gen(self):
        weight_orepa_origin = torch.einsum('oihw,o->oihw',
                                           self.weight_orepa_origin,
                                           self.vector[0, :])

        weight_orepa_avg = torch.einsum('oihw,hw->oihw', self.weight_orepa_avg_conv, self.weight_orepa_avg_avg)
        weight_orepa_avg = torch.einsum(
            'oihw,o->oihw',
            torch.einsum('oi,hw->oihw', self.weight_orepa_avg_conv.squeeze(3).squeeze(2),
                         self.weight_orepa_avg_avg), self.vector[1, :])

        weight_orepa_pfir = torch.einsum(
            'oihw,o->oihw',
            torch.einsum('oi,ohw->oihw', self.weight_orepa_pfir_conv.squeeze(3).squeeze(2),
                         self.weight_orepa_prior), self.vector[2, :])

        weight_orepa_1x1_kxk_conv1 = None
        if hasattr(self, 'weight_orepa_1x1_kxk_idconv1'):
            weight_orepa_1x1_kxk_conv1 = (self.weight_orepa_1x1_kxk_idconv1 +
                                          self.id_tensor).squeeze(3).squeeze(2)
        elif hasattr(self, 'weight_orepa_1x1_kxk_conv1'):
            weight_orepa_1x1_kxk_conv1 = self.weight_orepa_1x1_kxk_conv1.squeeze(3).squeeze(2)
        else:
            raise NotImplementedError
        weight_orepa_1x1_kxk_conv2 = self.weight_orepa_1x1_kxk_conv2

        if self.groups > 1:
            g = self.groups
            t, ig = weight_orepa_1x1_kxk_conv1.size()
            o, tg, h, w = weight_orepa_1x1_kxk_conv2.size()
            weight_orepa_1x1_kxk_conv1 = weight_orepa_1x1_kxk_conv1.view(
                g, int(t / g), ig)
            weight_orepa_1x1_kxk_conv2 = weight_orepa_1x1_kxk_conv2.view(
                g, int(o / g), tg, h, w)
            weight_orepa_1x1_kxk = torch.einsum('gti,gothw->goihw',
                                                weight_orepa_1x1_kxk_conv1,
                                                weight_orepa_1x1_kxk_conv2).reshape(
                o, ig, h, w)
        else:
            weight_orepa_1x1_kxk = torch.einsum('ti,othw->oihw',
                                                weight_orepa_1x1_kxk_conv1,
                                                weight_orepa_1x1_kxk_conv2)
        weight_orepa_1x1_kxk = torch.einsum('oihw,o->oihw', weight_orepa_1x1_kxk, self.vector[3, :])

        weight_orepa_1x1 = 0
        if hasattr(self, 'weight_orepa_1x1'):
            weight_orepa_1x1 = transVI_multiscale(self.weight_orepa_1x1,
                                                  self.kernel_size)
            weight_orepa_1x1 = torch.einsum('oihw,o->oihw', weight_orepa_1x1,
                                            self.vector[4, :])

        weight_orepa_gconv = self.dwsc2full(self.weight_orepa_gconv_dw,
                                            self.weight_orepa_gconv_pw,
                                            self.in_channels, self.groups)
        weight_orepa_gconv = torch.einsum('oihw,o->oihw', weight_orepa_gconv,
                                          self.vector[5, :])

        weight = weight_orepa_origin + weight_orepa_avg + weight_orepa_1x1 + weight_orepa_1x1_kxk + weight_orepa_pfir + weight_orepa_gconv

        return weight

    def dwsc2full(self, weight_dw, weight_pw, groups, groups_conv=1):

        t, ig, h, w = weight_dw.size()
        o, _, _, _ = weight_pw.size()
        tg = int(t / groups)
        i = int(ig * groups)
        ogc = int(o / groups_conv)
        groups_gc = int(groups / groups_conv)
        weight_dw = weight_dw.view(groups_conv, groups_gc, tg, ig, h, w)
        weight_pw = weight_pw.squeeze().view(ogc, groups_conv, groups_gc, tg)

        weight_dsc = torch.einsum('cgtihw,ocgt->cogihw', weight_dw, weight_pw)
        return weight_dsc.reshape(o, int(i / groups_conv), h, w)

    def forward(self, inputs=None):
        if hasattr(self, 'orepa_reparam'):
            return self.nonlinear(self.orepa_reparam(inputs))

        weight = self.weight_gen()

        if self.weight_only is True:
            return weight

        out = F.conv2d(
            inputs,
            weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)
        return self.nonlinear(self.bn(out))

    def get_equivalent_kernel_bias(self):
        return transI_fusebn(self.weight_gen(), self.bn)

    def switch_to_deploy(self):
        if hasattr(self, 'or1x1_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.orepa_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                       kernel_size=self.kernel_size, stride=self.stride,
                                       padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.orepa_reparam.weight.data = kernel
        self.orepa_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('weight_orepa_origin')
        self.__delattr__('weight_orepa_1x1')
        self.__delattr__('weight_orepa_1x1_kxk_conv2')
        if hasattr(self, 'weight_orepa_1x1_kxk_idconv1'):
            self.__delattr__('id_tensor')
            self.__delattr__('weight_orepa_1x1_kxk_idconv1')
        elif hasattr(self, 'weight_orepa_1x1_kxk_conv1'):
            self.__delattr__('weight_orepa_1x1_kxk_conv1')
        else:
            raise NotImplementedError
        self.__delattr__('weight_orepa_avg_avg')
        self.__delattr__('weight_orepa_avg_conv')
        self.__delattr__('weight_orepa_pfir_conv')
        self.__delattr__('weight_orepa_prior')
        self.__delattr__('weight_orepa_gconv_dw')
        self.__delattr__('weight_orepa_gconv_pw')

        self.__delattr__('bn')
        self.__delattr__('vector')

    def init_gamma(self, gamma_value):
        init.constant_(self.vector, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        init.constant_(self.vector[0, :], 1.0)


class OREPA_LargeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding=None, groups=1, dilation=1, act=True, deploy=False):
        super(OREPA_LargeConv, self).__init__()
        assert kernel_size % 2 == 1 and kernel_size > 3

        padding = myautopad(kernel_size, padding, dilation)
        self.stride = stride
        self.padding = padding
        self.layers = int((kernel_size - 1) / 2)
        self.groups = groups
        self.dilation = dilation

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        internal_channels = out_channels
        self.nonlinear = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        if deploy:
            self.or_large_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=kernel_size, stride=stride,
                                              padding=padding, dilation=dilation, groups=groups, bias=True)

        else:
            for i in range(self.layers):
                if i == 0:
                    self.__setattr__('weight' + str(i),
                                     OREPA(in_channels, internal_channels, kernel_size=3, stride=1, padding=1,
                                           groups=groups, weight_only=True))
                elif i == self.layers - 1:
                    self.__setattr__('weight' + str(i),
                                     OREPA(internal_channels, out_channels, kernel_size=3, stride=self.stride,
                                           padding=1, weight_only=True))
                else:
                    self.__setattr__('weight' + str(i),
                                     OREPA(internal_channels, internal_channels, kernel_size=3, stride=1, padding=1,
                                           weight_only=True))

            self.bn = nn.BatchNorm2d(out_channels)
            # self.unfold = torch.nn.Unfold(kernel_size=3, dilation=1, padding=2, stride=1)

    def weight_gen(self):
        weight = getattr(self, 'weight' + str(0)).weight_gen().transpose(0, 1)
        for i in range(self.layers - 1):
            weight2 = getattr(self, 'weight' + str(i + 1)).weight_gen()
            weight = F.conv2d(weight, weight2, groups=self.groups, padding=2)

        return weight.transpose(0, 1)
        '''
        weight = getattr(self, 'weight'+str(0))(inputs=None).transpose(0, 1)
        for i in range(self.layers - 1):
            weight = self.unfold(weight)
            weight2 = getattr(self, 'weight'+str(i+1))(inputs=None)

            weight = torch.einsum('akl,bk->abl', weight, weight2.view(weight2.size(0), -1))
            k = i * 2 + 5
            weight = weight.view(weight.size(0), weight.size(1), k, k)

        return weight.transpose(0, 1)
        '''

    def forward(self, inputs):
        if hasattr(self, 'or_large_reparam'):
            return self.nonlinear(self.or_large_reparam(inputs))

        weight = self.weight_gen()
        out = F.conv2d(inputs, weight, stride=self.stride, padding=self.padding, dilation=self.dilation,
                       groups=self.groups)
        return self.nonlinear(self.bn(out))

    def get_equivalent_kernel_bias(self):
        return transI_fusebn(self.weight_gen(), self.bn)

    def switch_to_deploy(self):
        if hasattr(self, 'or_large_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.or_large_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                          kernel_size=self.kernel_size, stride=self.stride,
                                          padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.or_large_reparam.weight.data = kernel
        self.or_large_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        for i in range(self.layers):
            self.__delattr__('weight' + str(i))
        self.__delattr__('bn')


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, deploy=False, nonlinear=None):
        super().__init__()
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        if deploy:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if hasattr(self, 'bn'):
            return self.nonlinear(self.bn(self.conv(x)))
        else:
            return self.nonlinear(self.conv(x))

    def switch_to_deploy(self):
        kernel, bias = transI_fusebn(self.conv.weight, self.bn)
        conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels,
                         kernel_size=self.conv.kernel_size,
                         stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation,
                         groups=self.conv.groups, bias=True)
        conv.weight.data = kernel
        conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv')
        self.__delattr__('bn')
        self.conv = conv


class OREPA_3x3_RepVGG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, groups=1, dilation=1, act=True,
                 internal_channels_1x1_3x3=None,
                 deploy=False):
        super(OREPA_3x3_RepVGG, self).__init__()
        self.deploy = deploy

        self.nonlinear = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        padding = myautopad(kernel_size, padding, dilation)
        assert padding == kernel_size // 2

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.branch_counter = 0

        self.weight_rbr_origin = nn.Parameter(
            torch.Tensor(out_channels, int(in_channels / self.groups), kernel_size, kernel_size))
        init.kaiming_uniform_(self.weight_rbr_origin, a=math.sqrt(1.0))
        self.branch_counter += 1

        if groups < out_channels:
            self.weight_rbr_avg_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            self.weight_rbr_pfir_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            init.kaiming_uniform_(self.weight_rbr_avg_conv, a=1.0)
            init.kaiming_uniform_(self.weight_rbr_pfir_conv, a=1.0)
            self.weight_rbr_avg_conv.data
            self.weight_rbr_pfir_conv.data
            self.register_buffer('weight_rbr_avg_avg',
                                 torch.ones(kernel_size, kernel_size).mul(1.0 / kernel_size / kernel_size))
            self.branch_counter += 1

        else:
            raise NotImplementedError
        self.branch_counter += 1

        if internal_channels_1x1_3x3 is None:
            internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels  # For mobilenet, it is better to have 2X internal channels

        if internal_channels_1x1_3x3 == in_channels:
            self.weight_rbr_1x1_kxk_idconv1 = nn.Parameter(
                torch.zeros(in_channels, int(in_channels / self.groups), 1, 1))
            id_value = np.zeros((in_channels, int(in_channels / self.groups), 1, 1))
            for i in range(in_channels):
                id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
            id_tensor = torch.from_numpy(id_value).type_as(self.weight_rbr_1x1_kxk_idconv1)
            self.register_buffer('id_tensor', id_tensor)

        else:
            self.weight_rbr_1x1_kxk_conv1 = nn.Parameter(
                torch.Tensor(internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1))
            init.kaiming_uniform_(self.weight_rbr_1x1_kxk_conv1, a=math.sqrt(1.0))
        self.weight_rbr_1x1_kxk_conv2 = nn.Parameter(
            torch.Tensor(out_channels, int(internal_channels_1x1_3x3 / self.groups), kernel_size, kernel_size))
        init.kaiming_uniform_(self.weight_rbr_1x1_kxk_conv2, a=math.sqrt(1.0))
        self.branch_counter += 1

        expand_ratio = 8
        self.weight_rbr_gconv_dw = nn.Parameter(torch.Tensor(in_channels * expand_ratio, 1, kernel_size, kernel_size))
        self.weight_rbr_gconv_pw = nn.Parameter(torch.Tensor(out_channels, in_channels * expand_ratio, 1, 1))
        init.kaiming_uniform_(self.weight_rbr_gconv_dw, a=math.sqrt(1.0))
        init.kaiming_uniform_(self.weight_rbr_gconv_pw, a=math.sqrt(1.0))
        self.branch_counter += 1

        if out_channels == in_channels and stride == 1:
            self.branch_counter += 1

        self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
        self.bn = nn.BatchNorm2d(out_channels)

        self.fre_init()

        init.constant_(self.vector[0, :], 0.25)  # origin
        init.constant_(self.vector[1, :], 0.25)  # avg
        init.constant_(self.vector[2, :], 0.0)  # prior
        init.constant_(self.vector[3, :], 0.5)  # 1x1_kxk
        init.constant_(self.vector[4, :], 0.5)  # dws_conv

    def fre_init(self):
        prior_tensor = torch.Tensor(self.out_channels, self.kernel_size, self.kernel_size)
        half_fg = self.out_channels / 2
        for i in range(self.out_channels):
            for h in range(3):
                for w in range(3):
                    if i < half_fg:
                        prior_tensor[i, h, w] = math.cos(math.pi * (h + 0.5) * (i + 1) / 3)
                    else:
                        prior_tensor[i, h, w] = math.cos(math.pi * (w + 0.5) * (i + 1 - half_fg) / 3)

        self.register_buffer('weight_rbr_prior', prior_tensor)

    def weight_gen(self):

        weight_rbr_origin = torch.einsum('oihw,o->oihw', self.weight_rbr_origin, self.vector[0, :])

        weight_rbr_avg = torch.einsum('oihw,o->oihw',
                                      torch.einsum('oihw,hw->oihw', self.weight_rbr_avg_conv, self.weight_rbr_avg_avg),
                                      self.vector[1, :])

        weight_rbr_pfir = torch.einsum('oihw,o->oihw',
                                       torch.einsum('oihw,ohw->oihw', self.weight_rbr_pfir_conv, self.weight_rbr_prior),
                                       self.vector[2, :])

        weight_rbr_1x1_kxk_conv1 = None
        if hasattr(self, 'weight_rbr_1x1_kxk_idconv1'):
            weight_rbr_1x1_kxk_conv1 = (self.weight_rbr_1x1_kxk_idconv1 + self.id_tensor).squeeze()
        elif hasattr(self, 'weight_rbr_1x1_kxk_conv1'):
            weight_rbr_1x1_kxk_conv1 = self.weight_rbr_1x1_kxk_conv1.squeeze()
        else:
            raise NotImplementedError
        weight_rbr_1x1_kxk_conv2 = self.weight_rbr_1x1_kxk_conv2

        if self.groups > 1:
            g = self.groups
            t, ig = weight_rbr_1x1_kxk_conv1.size()
            o, tg, h, w = weight_rbr_1x1_kxk_conv2.size()
            weight_rbr_1x1_kxk_conv1 = weight_rbr_1x1_kxk_conv1.view(g, int(t / g), ig)
            weight_rbr_1x1_kxk_conv2 = weight_rbr_1x1_kxk_conv2.view(g, int(o / g), tg, h, w)
            weight_rbr_1x1_kxk = torch.einsum('gti,gothw->goihw', weight_rbr_1x1_kxk_conv1,
                                              weight_rbr_1x1_kxk_conv2).view(o, ig, h, w)
        else:
            weight_rbr_1x1_kxk = torch.einsum('ti,othw->oihw', weight_rbr_1x1_kxk_conv1, weight_rbr_1x1_kxk_conv2)

        weight_rbr_1x1_kxk = torch.einsum('oihw,o->oihw', weight_rbr_1x1_kxk, self.vector[3, :])

        weight_rbr_gconv = self.dwsc2full(self.weight_rbr_gconv_dw, self.weight_rbr_gconv_pw, self.in_channels)
        weight_rbr_gconv = torch.einsum('oihw,o->oihw', weight_rbr_gconv, self.vector[4, :])

        weight = weight_rbr_origin + weight_rbr_avg + weight_rbr_1x1_kxk + weight_rbr_pfir + weight_rbr_gconv

        return weight

    def dwsc2full(self, weight_dw, weight_pw, groups):

        t, ig, h, w = weight_dw.size()
        o, _, _, _ = weight_pw.size()
        tg = int(t / groups)
        i = int(ig * groups)
        weight_dw = weight_dw.view(groups, tg, ig, h, w)
        weight_pw = weight_pw.squeeze().view(o, groups, tg)

        weight_dsc = torch.einsum('gtihw,ogt->ogihw', weight_dw, weight_pw)
        return weight_dsc.view(o, i, h, w)

    def forward(self, inputs):
        weight = self.weight_gen()
        out = F.conv2d(inputs, weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation,
                       groups=self.groups)

        return self.nonlinear(self.bn(out))


class Bottleneck_OREPA(Bottleneck):
    """Standard bottleneck with OREPA."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = OREPA(c1, c_)
        self.cv2 = OREPA(c_, c2, groups=g)


class C3_OREPA(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_OREPA(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


# ---------------------  C3_OREPA end--------------------------------------

######################################## EfficientNet-MBConv start ########################################

class MBConv(nn.Module):
    def __init__(self, inc, ouc, shortcut=True, e=4, dropout=0.1) -> None:
        super().__init__()
        midc = inc * e
        self.conv_pw_1 = Conv(inc, midc, 1)
        self.conv_dw_1 = Conv(midc, midc, 3, g=midc)
        self.effective_se = EffectiveSEModule(midc)
        self.conv1 = Conv(midc, ouc, 1, act=False)
        self.dropout = nn.Dropout2d(p=dropout)
        self.add = shortcut and inc == ouc
    
    def forward(self, x):
        return x + self.dropout(self.conv1(self.effective_se(self.conv_dw_1(self.conv_pw_1(x))))) if self.add else self.dropout(self.conv1(self.effective_se(self.conv_dw_1(self.conv_pw_1(x)))))

class C3_EMBC(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MBConv(c_, c_, shortcut) for _ in range(n)))

############################################################上采样##############################################################################

#----------------------------------------上采样 CARAFE start--------------------------------------------------------------------------------------
class CARAFE(nn.Module):
    # CARAFE: Content-Aware ReAssembly of FEatures       https://arxiv.org/pdf/1905.02188.pdf
    def __init__(self, c1, c2, kernel_size=5, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = 5
        self.up_factor = 2
        self.down = nn.Conv2d(c1, c1 // 4, 1)
        self.encoder = nn.Conv2d(c1 // 4, self.up_factor ** 2 * self.kernel_size ** 2, self.kernel_size, 1,
                                 self.kernel_size // 2)
        self.out = nn.Conv2d(c1, c2, 1)
 
    def forward(self, x):
        N, C, H, W = x.size()
        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W,
                                              self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)
 
        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        x = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2),
                  mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        x = x.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        x = x.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        x = x.reshape(N, C, H, W, -1)  # (N, C, H, W, Kup^2)
        x = x.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)
 
        out_tensor = torch.matmul(x, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        # print("up shape:",out_tensor.shape)
        return out_tensor
#----------------------------------------上采样 CARAFE end-----------------------------------------------

#-------------------------------------FFCA Yolo IEEE2024 start---------------------------------------------------
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out

class FFM_Concat2(nn.Module):
    def __init__(self, dimension=1, Channel1 = 1, Channel2 = 1):
        super(FFM_Concat2, self).__init__()
        self.d = dimension
        self.Channel1 = Channel1
        self.Channel2 = Channel2
        self.Channel_all = int(Channel1 + Channel2)
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型 parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化

    def forward(self, x):
        N1, C1, H1, W1 = x[0].size()
        N2, C2, H2, W2 = x[1].size()

        w = self.w[:(C1 + C2)] # 加了这一行可以确保能够剪枝
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion

        x1 = (weight[:C1] * x[0].view(N1, H1, W1, C1)).view(N1, C1, H1, W1)
        x2 = (weight[C1:] * x[1].view(N2, H2, W2, C2)).view(N2, C2, H2, W2)
        x = [x1, x2]
        return torch.cat(x, self.d)

class FFM_Concat3(nn.Module):
    def __init__(self, dimension=1, Channel1 = 1, Channel2 = 1, Channel3 = 1):
        super(FFM_Concat3, self).__init__()
        self.d = dimension
        self.Channel1 = Channel1
        self.Channel2 = Channel2
        self.Channel3 = Channel3
        self.Channel_all = int(Channel1 + Channel2 + Channel3)
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        N1, C1, H1, W1 = x[0].size()
        N2, C2, H2, W2 = x[1].size()
        N3, C3, H3, W3 = x[2].size()

        w = self.w[:(C1 + C2 + C3)]  # 加了这一行可以确保能够剪枝
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion

        x1 = (weight[:C1] * x[0].view(N1, H1, W1, C1)).view(N1, C1, H1, W1)
        x2 = (weight[C1:(C1 + C2)] * x[1].view(N2, H2, W2, C2)).view(N2, C2, H2, W2)
        x3 = (weight[(C1 + C2):] * x[2].view(N3, H3, W3, C3)).view(N3, C3, H3, W3)
        x = [x1, x2, x3]
        return torch.cat(x, self.d)

def autopad_FFCA(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv_withoutBN(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad_FFCA(k, p, d), groups=g, dilation=d, bias=False)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))

class SCAM(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super(SCAM, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels

        self.k = Conv(in_channels, 1, 1, 1)
        self.v = Conv(in_channels, self.inter_channels, 1, 1)
        self.m = Conv_withoutBN(self.inter_channels, in_channels, 1, 1)
        self.m2 = Conv(2, 1, 1, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # GMP

    def forward(self, x):
        n, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)

        # avg max: [N, C, 1, 1]
        avg = self.avg_pool(x).softmax(1).view(n, 1, 1, c)
        max = self.max_pool(x).softmax(1).view(n, 1, 1, c)

        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)

        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)

        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)

        # y2:[N, 1, H, W]
        y_avg = torch.matmul(avg, v).view(n, 1, h, w)
        y_max = torch.matmul(max, v).view(n, 1, h, w)

        # y_cat:[N, 2, H, W]
        y_cat = torch.cat((y_avg, y_max), 1)

        y = self.m(y) * self.m2(y_cat).sigmoid()

        return x + y
#-------------------------------------FFCA Yolo IEEE2024 end---------------------------------------------------

######################################## FocalModulation start ########################################
class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window=3, focal_level=2, focal_factor=2, bias=True, proj_drop=0., use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f_linear = nn.Conv2d(dim, 2 * dim + (self.focal_level + 1), kernel_size=1, bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()
                
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, 
                    groups=dim, padding=kernel_size//2, bias=False),
                    nn.GELU(),
                    )
                )              
            self.kernel_sizes.append(kernel_size)          
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[1]

        # pre linear projection
        x = self.f_linear(x).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level+1), 1)
        
        # context aggreation
        ctx_all = 0.0
        for l in range(self.focal_level):         
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        x_out = q * self.h(ctx_all)
        x_out = x_out.contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)
        
        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

######################################## FocalModulation end ########################################