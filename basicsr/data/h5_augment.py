import torch
import torch.nn.functional as F
import torchvision.transforms
from math import sin, cos, pi
import numbers
import numpy as np
import random


"""
    Data augmentation functions.

    heavily borrowed from https://github.com/TimoStoff/event_utils

    @InProceedings{Stoffregen19cvpr,
    author = {Stoffregen, Timo and Kleeman, Lindsay},
    title = {Event Cameras, Contrast Maximization and Reward Functions: An Analysis},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
    } 
    
    There are some problems with torchvision data augmentation functions:
    1. they work only on PIL images, which means they cannot be applied to tensors with more than 3 channels,
       and they require a lot of conversion from Numpy -> PIL -> Tensor

    2. they do not provide access to the internal transformations (affine matrices) used, which prevent
       applying them for more complex tasks, such as transformation of an optic flow field (for which
       the inverse transformation must be known).

    For these reasons, we implement my own data augmentation functions
    (strongly inspired by torchvision transforms) that operate directly
    on Torch Tensor variables, and that allow to transform an optic flow field as well.
"""


def normalize_image_sequence_(sequence, key='frame'):
    images = torch.stack([item[key] for item in sequence], dim=0)
    mini = np.percentile(torch.flatten(images), 1)
    maxi = np.percentile(torch.flatten(images), 99)
    images = (images - mini) / (maxi - mini + 1e-5)
    images = torch.clamp(images, 0, 1)
    for i, item in enumerate(sequence):
        item[key] = images[i, ...]


def put_hot_pixels_in_voxel_(voxel, hot_pixel_range=1.0, hot_pixel_fraction=0.001):
    num_hot_pixels = int(hot_pixel_fraction * voxel.shape[-1] * voxel.shape[-2])
    x = torch.randint(0, voxel.shape[-1], (num_hot_pixels,))
    y = torch.randint(0, voxel.shape[-2], (num_hot_pixels,))
    for i in range(num_hot_pixels):
        voxel[..., :, y[i], x[i]] = random.uniform(-hot_pixel_range, hot_pixel_range)


def add_hot_pixels_to_sequence_(sequence, hot_pixel_std=1.0, max_hot_pixel_fraction=0.001):
    hot_pixel_fraction = random.uniform(0, max_hot_pixel_fraction)
    voxel = sequence[0]['events']
    num_hot_pixels = int(hot_pixel_fraction * voxel.shape[-1] * voxel.shape[-2])
    x = torch.randint(0, voxel.shape[-1], (num_hot_pixels,))
    y = torch.randint(0, voxel.shape[-2], (num_hot_pixels,))
    val = torch.randn(num_hot_pixels, dtype=voxel.dtype, device=voxel.device)
    val *= hot_pixel_std
    # TODO multiprocessing
    for item in sequence:
        for i in range(num_hot_pixels):
            item['events'][..., :, y[i], x[i]] += val[i]


def add_noise_to_voxel(voxel, noise_std=1.0, noise_fraction=0.1):
    noise = noise_std * torch.randn_like(voxel)  # mean = 0, std = noise_std
    if noise_fraction < 1.0:
        mask = torch.rand_like(voxel) >= noise_fraction
        noise.masked_fill_(mask, 0)
    return voxel + noise


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> torchvision.transforms.Compose([
        >>>     torchvision.transforms.CenterCrop(10),
        >>>     torchvision.transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, is_flow=False):
        for t in self.transforms:
            x = t(x, is_flow)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class CenterCrop(object):
    """Center crop the tensor to a certain size.
    """

    def __init__(self, size, preserve_mosaicing_pattern=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.preserve_mosaicing_pattern = preserve_mosaicing_pattern

    def __call__(self, x, is_flow=False):
        """
            x: [C x H x W] Tensor to be rotated.
            is_flow: this parameter does not have any effect
        Returns:
            Tensor: Cropped tensor.
        """
        w, h = x.shape[2], x.shape[1]
        th, tw = self.size
        assert (th <= h)
        assert (tw <= w)
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        if self.preserve_mosaicing_pattern:
            # make sure that i and j are even, to preserve
            # the mosaicing pattern
            if i % 2 == 1:
                i = i + 1
            if j % 2 == 1:
                j = j + 1

        return x[:, i:i + th, j:j + tw]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RobustNorm(object):
    """
    Robustly normalize tensor
    """

    def __init__(self, low_perc=0, top_perc=95):
        self.top_perc = top_perc
        self.low_perc = low_perc

    @staticmethod
    def percentile(t, q):
        """
        Return the ``q``-th percentile of the flattened input tensor's data.

        CAUTION:
         * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
         * Values are not interpolated, which corresponds to
           ``numpy.percentile(..., interpolation="nearest")``.

        :param t: Input tensor.
        :param q: Percentile to compute, which must be between 0 and 100 inclusive.
        :return: Resulting value (scalar).
        """
        # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
        # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
        # so that ``round()`` returns an integer, even if q is a np.float32.
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        try:
            result = t.view(-1).kthvalue(k).values.item()
        except RuntimeError:
            result = t.reshape(-1).kthvalue(k).values.item()
        return result

    def __call__(self, x, is_flow=False):
        """
        """
        t_max = self.percentile(x, self.top_perc)
        t_min = self.percentile(x, self.low_perc)
        # print("t_max={}, t_min={}".format(t_max, t_min))
        if t_max == 0 and t_min == 0:
            return x
        eps = 1e-6
        normed = torch.clamp(x, min=t_min, max=t_max)
        normed = (normed - torch.min(normed)) / (torch.max(normed) + eps)
        return normed

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += '(top_perc={:.2f}'.format(self.top_perc)
        format_string += ', low_perc={:.2f})'.format(self.low_perc)
        return format_string


class LegacyNorm(object):
    """
    Rescale tensor to mean=0 and standard deviation std=1
    """

    def __call__(self, x, is_flow=False):
        """
        Compute mean and stddev of the **nonzero** elements of the event tensor
        we do not use PyTorch's default mean() and std() functions since it's faster
        to compute it by hand than applying those funcs to a masked array
        """
        nonzero = (x != 0)
        num_nonzeros = nonzero.sum()
        if num_nonzeros > 0:
            mean = x.sum() / num_nonzeros
            stddev = torch.sqrt((x ** 2).sum() / num_nonzeros - mean ** 2)
            mask = nonzero.float()
            x = mask * (x - mean) / stddev
        return x

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string



class RandomCrop(object):
    """Crop the tensor at a random location.
    """

    def __init__(self, size, preserve_mosaicing_pattern=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.preserve_mosaicing_pattern = preserve_mosaicing_pattern

    @staticmethod
    def get_params(x, output_size):
        w, h = x.shape[-1], x.shape[-2]
        th, tw = output_size
        if th > h or tw > w:
            raise Exception("Input size {}x{} is less than desired cropped \
                    size {}x{} - input tensor shape = {}".format(w, h, tw, th, x.shape))
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return i, j, th, tw

    def __call__(self, x, is_flow=False):
        """
            x: [C x H x W] Tensor to be rotated.
            is_flow: this parameter does not have any effect
        Returns:
            Tensor: Cropped tensor.
        """
        i, j, h, w = self.get_params(x, self.size)

        if self.preserve_mosaicing_pattern:
            # make sure that i and j are even, to preserve the mosaicing pattern
            if i % 2 == 1:
                i = i + 1
            if j % 2 == 1:
                j = j + 1

        return x[:, i:i + h, j:j + w]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomRotationFlip(object):
    """Rotate the image by angle.
    """

    def __init__(self, degrees, p_hflip=0.5, p_vflip=0.5):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    @staticmethod
    def get_params(degrees, p_hflip, p_vflip):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])
        angle_rad = angle * pi / 180.0

        M_original_transformed = torch.FloatTensor([[cos(angle_rad), -sin(angle_rad), 0],
                                                    [sin(angle_rad), cos(angle_rad), 0],
                                                    [0, 0, 1]])

        if random.random() < p_hflip:
            M_original_transformed[:, 0] *= -1

        if random.random() < p_vflip:
            M_original_transformed[:, 1] *= -1

        M_transformed_original = torch.inverse(M_original_transformed)

        M_original_transformed = M_original_transformed[:2, :].unsqueeze(dim=0)  # 3 x 3 -> N x 2 x 3
        M_transformed_original = M_transformed_original[:2, :].unsqueeze(dim=0)

        return M_original_transformed, M_transformed_original

    def __call__(self, x, is_flow=False):
        """
            x: [C x H x W] Tensor to be rotated.
            is_flow: if True, x is an [2 x H x W] displacement field, which will also be transformed
        Returns:
            Tensor: Rotated tensor.
        """
        assert (len(x.shape) == 3)

        if is_flow:
            assert (x.shape[0] == 2)

        M_original_transformed, M_transformed_original = self.get_params(self.degrees, self.p_hflip, self.p_vflip)
        affine_grid = F.affine_grid(M_original_transformed, x.unsqueeze(dim=0).shape)
        transformed = F.grid_sample(x.unsqueeze(dim=0), affine_grid)

        if is_flow:
            # Apply the same transformation to the flow field
            A00 = M_transformed_original[0, 0, 0]
            A01 = M_transformed_original[0, 0, 1]
            A10 = M_transformed_original[0, 1, 0]
            A11 = M_transformed_original[0, 1, 1]
            vx = transformed[:, 0, :, :].clone()
            vy = transformed[:, 1, :, :].clone()
            transformed[:, 0, :, :] = A00 * vx + A01 * vy
            transformed[:, 1, :, :] = A10 * vx + A11 * vy

        return transformed.squeeze(dim=0)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', p_flip={:.2f}'.format(self.p_hflip)
        format_string += ', p_vlip={:.2f}'.format(self.p_vflip)
        format_string += ')'
        return format_string


class RandomFlip(object):
    """
    Flip tensor along last two dims
    """

    def __init__(self, p_hflip=0.5, p_vflip=0.5):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    def __call__(self, x, is_flow=False):
        """
        :param x: [... x H x W] Tensor to be flipped.
        :param is_flow: if True, x is an [... x 2 x H x W] displacement field, which will also be transformed
        :return Tensor: Flipped tensor.
        """
        assert (len(x.shape) >= 2)
        if is_flow:
            assert (len(x.shape) >= 3)
            assert (x.shape[-3] == 2)

        dims = []
        self.ran = random.random() 
        if self.ran < self.p_hflip:
            dims.append(-1)

        if self.ran < self.p_vflip:
            dims.append(-2)

        if not dims:
            return x

        flipped = torch.flip(x, dims=dims)
        if is_flow:
            for d in dims:
                idx = -(d + 1)  # swap since flow is x, y
                flipped[..., idx, :, :] *= -1
        return flipped


    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += '(p_flip={:.2f}'.format(self.p_hflip)
        format_string += ', p_vlip={:.2f})'.format(self.p_vflip)
        format_string += ', random={:.2f})'.format(self.ran)
        return format_string
