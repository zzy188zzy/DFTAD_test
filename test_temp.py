import torch
from libs.modeling.deformable_trans import DeformableTransformer

if __name__ == "__main__":
    device = "cuda"
    model = DeformableTransformer(feature_dimm=512).to(device)
    data_x = torch.randn([2, 512, 2304], device=device)
    data_masks = torch.ones([2, 2304], device=device).bool()
    data_masks[..., 1999:] = False
    x, mask = model(data_x, data_masks)
    print(x.shape, mask.shape)

