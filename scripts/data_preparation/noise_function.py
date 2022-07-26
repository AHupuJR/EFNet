
def put_hot_pixels_in_voxel_(voxel, hot_pixel_range=20, hot_pixel_fraction=0.00002):
    num_hot_pixels = int(hot_pixel_fraction * voxel.shape[-1] * voxel.shape[-2])
    x = torch.randint(0, voxel.shape[-1], (num_hot_pixels,))
    y = torch.randint(0, voxel.shape[-2], (num_hot_pixels,))
    for i in range(num_hot_pixels):
        # voxel[..., :, y[i], x[i]] = random.uniform(-hot_pixel_range, hot_pixel_range)
        voxel[..., :, y[i], x[i]] = random.randint(-hot_pixel_range, hot_pixel_range)


def add_noise_to_voxel(voxel, noise_std=1.0, noise_fraction=0.1):
    noise = noise_std * torch.randn_like(voxel)  # mean = 0, std = noise_std
    if noise_fraction < 1.0:
        mask = torch.rand_like(voxel) >= noise_fraction
        noise.masked_fill_(mask, 0)
    return voxel + noise
