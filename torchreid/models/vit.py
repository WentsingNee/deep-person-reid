
import timm

__all__ = [
    'vit'
]

def vit(num_classes=1000, **kwargs):
    # Load pre-trained ViT
    vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    return vit_base
