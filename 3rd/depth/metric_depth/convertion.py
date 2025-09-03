import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from IPython.display import display
import torch
from PIL import Image
import coremltools as ct
from huggingface_hub import hf_hub_download
import tempfile
import matplotlib
import numpy as np
from torch import nn
from depth_anything_v2.dpt import DepthAnythingV2


from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil import Builder as mb

@register_torch_op
def upsample_bicubic2d(context, node):
    inputs = _get_inputs(context, node)

    for input in inputs:
        print("DBS upsample_bicubic2d, input: ", input)
        if hasattr(input, 'name'):
            print("DBS upsample_bicubic2d, input name: ", input.name)
            if (input.name == '170' or input.name == '173') and hasattr(input, 'val'):
                print("DBS upsample_bicubic2d, input value: ", input.val)
        if hasattr(input, 'shape'):
            print("DBS upsample_bicubic2d, input shape: ", input.shape)
        if hasattr(input, 'dtype'):
            print("DBS upsample_bicubic2d, input dtype: ", input.dtype)
        if hasattr(input, 'type_str'):
            print("DBS upsample_bicubic2d, input type_str: ", input.type_str)
        print("\n")

    a = inputs[0]
    b = inputs[3]
    print("DBS upsample_bicubic2d, a: ", a)
    print("DBS upsample_bicubic2d, b: ", b)
    y = mb.resize_bilinear(
        x=a, 
        target_size_height=int(b.val[0] * a.shape[2]), 
        target_size_width=int(b.val[1] * a.shape[3]), 
        name=node.name
    )
    context.add(y)

class DepthModelWrapper(nn.Module):
    def __init__(self, model):
        super(DepthModelWrapper, self).__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # return self.model(x)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        output = self.model(x)  # output shape: [B, H, W]

        # 扩展出通道维度 → [B, 1, H, W]
        output_exp = output.unsqueeze(1)

        # 前三个通道复制原始值 → [B, 3, H, W]
        repeated = output_exp.repeat(1, 3, 1, 1)

        # 归一化每个样本的 output → [B, 1, H, W]
        min_vals = output.amin(dim=(1, 2), keepdim=True)
        max_vals = output.amax(dim=(1, 2), keepdim=True)
        normed = (output - min_vals) / (max_vals - min_vals + 1e-6)
        normed = normed.unsqueeze(1)  # → [B, 1, H, W]

        # 拼接：前3通道 + 最后1通道 → [B, 4, H, W]
        out4 = torch.cat([repeated, normed], dim=1)

        return out4.permute(0, 2, 3, 1)
        # return output
    

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
encoder2name = {
    'vits': 'Small',
    'vitb': 'Base',
    'vitl': 'Large',
    'vitg': 'Giant', # UNAVAILABLE AS OF TODAY
}
encoder = 'vitl'
model_name = encoder2name[encoder]
model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': 20})
# filepath = hf_hub_download(repo_id=f"depth-anything/Depth-Anything-V2-{model_name}", filename=f"depth_anything_v2_{encoder}.pth", repo_type="model")
state_dict = torch.load('/Users/qhong/Documents/MM/motionmodel/3rd/depth/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth', map_location="cpu")
model.load_state_dict(state_dict)
# checkpoint = torch.load('/Users/qhong/Documents/MM/motionmodel/3rd/depth/metric_depth/checkpoints/DA2-metric-vitb-ue-ep8-MD1000-R1918_1008.pth', map_location='cpu')['model']
# tmp = {k.replace('module.',''): v for k,v in checkpoint.items()}
# model.load_state_dict(tmp)
model = model.to(device).eval()
model_wrapper = DepthModelWrapper(model).eval()
input_tensor = torch.randn(1, 3, 518, 924).to(device)
# input_shape = (1, 3, ct.RangeDim(upper_bound=1008), ct.RangeDim(upper_bound=1918))

traceable_model = torch.jit.trace(model_wrapper, input_tensor)

# Set the image scale and bias for input image preprocessing
scale = 1 / (0.226 * 255.0)
bias = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]

# Define the input and output types for the CoreML model
input_name = "input"
output_name = "output"
input_type = ct.ImageType(name=input_name, shape=input_tensor.shape, scale=scale, bias=bias,
                          color_layout=ct.colorlayout.RGB)
output_type = ct.ImageType(name=output_name, color_layout=ct.colorlayout.GRAYSCALE)

mlmodel = ct.convert(
    traceable_model,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT32,
    compute_units=ct.ComputeUnit.CPU_AND_GPU,
    inputs=[ct.TensorType(name="input",shape=input_tensor.shape, dtype=np.float32)],
    # inputs = [input_type],
    outputs=[
        ct.TensorType(name="output", dtype=np.float32),
    ],
)

# mlmodel.save(f"DepthAnythingV2{model_name}.mlpackage")
coreml_model_file_name = "Da2.mlpackage"
print(f"Saving model to {coreml_model_file_name}")
mlmodel.save(coreml_model_file_name)
print("Done!")