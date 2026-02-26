"""Conv1d → Conv2d post-load conversion for NHWC layout elimination.

Conv1d operates on 3D tensors (N,C,L) which don't support channels_last.
cuDNN inserts nchwToNhwcKernel/nhwcToNchwKernel on every Conv1d call.
By converting Conv1d → Conv2d with kernel (1,k), input becomes 4D (N,C,1,L)
which supports channels_last, eliminating ~257 layout conversion kernels.

Usage (after checkpoint load, before enable_channels_last):
    from unifolm_wma.ops.wrappers.conv1d_to_conv2d import enable_conv2d_nhwc
    n = enable_conv2d_nhwc(model.action_unet)
"""
import torch.nn as nn


def _conv1d_to_conv2d(conv1d: nn.Conv1d) -> nn.Conv2d:
    """Conv1d(in, out, k, s, p) → Conv2d(in, out, (1,k), (1,s), (0,p))"""
    conv2d = nn.Conv2d(
        conv1d.in_channels,
        conv1d.out_channels,
        kernel_size=(1, conv1d.kernel_size[0]),
        stride=(1, conv1d.stride[0]),
        padding=(0, conv1d.padding[0]),
        dilation=(1, conv1d.dilation[0]),
        groups=conv1d.groups,
        bias=conv1d.bias is not None,
        padding_mode=conv1d.padding_mode,
    )
    # weight: (out, in/g, k) → (out, in/g, 1, k)
    conv2d.weight.data = conv1d.weight.data.unsqueeze(2)
    if conv1d.bias is not None:
        conv2d.bias.data = conv1d.bias.data.clone()
    return conv2d


def _convtranspose1d_to_convtranspose2d(ct1d: nn.ConvTranspose1d) -> nn.ConvTranspose2d:
    """ConvTranspose1d(in, out, k, s, p) → ConvTranspose2d(in, out, (1,k), (1,s), (0,p))"""
    ct2d = nn.ConvTranspose2d(
        ct1d.in_channels,
        ct1d.out_channels,
        kernel_size=(1, ct1d.kernel_size[0]),
        stride=(1, ct1d.stride[0]),
        padding=(0, ct1d.padding[0]),
        output_padding=(0, ct1d.output_padding[0]),
        groups=ct1d.groups,
        bias=ct1d.bias is not None,
        dilation=(1, ct1d.dilation[0]),
        padding_mode=ct1d.padding_mode,
    )
    # weight: (in, out/g, k) → (in, out/g, 1, k)
    ct2d.weight.data = ct1d.weight.data.unsqueeze(2)
    if ct1d.bias is not None:
        ct2d.bias.data = ct1d.bias.data.clone()
    return ct2d


def enable_conv2d_nhwc(module: nn.Module) -> int:
    """Replace all Conv1d/ConvTranspose1d with Conv2d/ConvTranspose2d equivalents.

    Traverses the module tree and swaps in-place. Sets module._use_conv2d = True
    so forward methods can unsqueeze/squeeze appropriately.

    Returns the number of modules replaced.
    """
    count = 0
    for name, child in list(module.named_modules()):
        if isinstance(child, nn.Conv1d):
            new_mod = _conv1d_to_conv2d(child)
            _set_submodule(module, name, new_mod)
            count += 1
        elif isinstance(child, nn.ConvTranspose1d):
            new_mod = _convtranspose1d_to_convtranspose2d(child)
            _set_submodule(module, name, new_mod)
            count += 1

    module._use_conv2d = True
    return count


def _set_submodule(root: nn.Module, name: str, new_module: nn.Module):
    """Set a named submodule, handling both attribute and Sequential index access."""
    parts = name.rsplit('.', 1)
    if len(parts) > 1:
        parent = root.get_submodule(parts[0])
        attr = parts[1]
    else:
        parent = root
        attr = name

    if isinstance(parent, nn.Sequential) and attr.isdigit():
        parent[int(attr)] = new_module
    else:
        setattr(parent, attr, new_module)
