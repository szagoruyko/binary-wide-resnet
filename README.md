1-bit Wide ResNet
===========

PyTorch implementation of training 1-bit Wide ResNets from this paper:

*Training wide residual networks for deployment using a single bit for each weight* by **Mark D. McDonnell** at ICLR 2018

<https://openreview.net/forum?id=rytNfI1AZ>

<https://arxiv.org/abs/1802.08530>

The idea is very simple but surprisingly effective for training ResNets with binary weights. Here is the proposed weight parameterization as PyTorch autograd function:

```python
class ForwardSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        return math.sqrt(2. / (w.shape[1] * w.shape[2] * w.shape[3])) * w.sign()

    @staticmethod
    def backward(ctx, g):
        return g
```

On forward, we take sign of the weights and scale it by He-init constant. On backward, we propagate gradient without changes. WRN-20-10 trained with such parameterization is only slightly off from it's full precision variant, here is what I got myself with this code on CIFAR-100:

| network | accuracy (5 runs mean +- std) | checkpoint (Mb) |
|:---|:---:|:---:|
| WRN-20-10 | 80.5 +- 0.24 | 205 Mb |
| WRN-20-10-1bit | 80.0 +- 0.26 | 3.5 Mb |

## Details

Here are the differences with WRN code <https://github.com/szagoruyko/wide-residual-networks>:

* BatchNorm has no affine weight and bias parameters
* First layer has 16 * width channels
* Last fc layer is removed in favor of 1x1 conv + F.avg_pool2d
* Downsample is done by F.avg_pool2d + torch.cat instead of strided conv
* SGD with cosine annealing and warm restarts

I used PyTorch 0.4.1 and Python 3.6 to run the code.

Reproduce WRN-20-10 with 1-bit training on CIFAR-100:

```bash
python main.py --binarize --save ./logs/WRN-20-10-1bit_$RANDOM --width 10 --dataset CIFAR100
```

Convergence plot (train error in dash):

<img width="950" alt="download" src="https://user-images.githubusercontent.com/4953728/44685365-968ea500-aa4b-11e8-8615-684120f13953.png">

I've also put 3.5 Mb checkpoint with binary weights packed with `np.packbits`, and a very short script to evaluate it:

```bash
python evaluate_packed.py --checkpoint wrn20-10-1bit-packed.pth.tar --width 10 --dataset CIFAR100
```

S3 url to checkpoint: <https://s3.amazonaws.com/modelzoo-networks/wrn20-10-1bit-packed.pth.tar>
