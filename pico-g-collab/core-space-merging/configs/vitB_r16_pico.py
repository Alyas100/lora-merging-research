import os

PICO_ROOT = '/content/lora-merging-research/pico'
MODEL_DIR = os.path.join(PICO_ROOT, 'KnOTS-ViT-B-32-lora-R16-adapters')
CACHE_DIR = os.path.join(PICO_ROOT, 'data')
HEAD_DIR = os.path.join(PICO_ROOT, 'ViT-B-32-CLIP')

config = {
    'dataset': [
        {
            'name': 'gtsrb',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, 'gtsrb_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 16,
            'shuffled_idxs': os.path.join(PICO_ROOT, 'core-space-merging/dataset/shuffled_idxs/gtsrb_shuffled_idxs.pt'),
        },
        {
            'name': 'mnist',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, 'mnist_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,
            'shuffled_idxs': os.path.join(PICO_ROOT, 'core-space-merging/dataset/shuffled_idxs/mnist_shuffled_idxs.pt'),
        },
        {
            'name': 'svhn',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, 'svhn_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,
            'shuffled_idxs': os.path.join(PICO_ROOT, 'core-space-merging/dataset/shuffled_idxs/svhn_shuffled_idxs.pt'),
        },
    ],
    'model': {
        'name': 'hf_clip',
        'base_type': 'openai/clip-vit-base-patch32',
        'cachedir': os.path.join(PICO_ROOT, 'ViT-B-32-CLIP'),
        'bases': [
            os.path.join(MODEL_DIR, 'KnOTS-ViT-B-32_lora_R16_gtsrb'),
            os.path.join(MODEL_DIR, 'KnOTS-ViT-B-32_lora_R16_mnist'),
            os.path.join(MODEL_DIR, 'KnOTS-ViT-B-32_lora_R16_svhn'),
        ],
        'ft_config': {
            'type': 'lora',
            'r': 16,
            'lora_alpha': 16,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
            'lora_dropout': 0.1,
            'bias': 'none',
        },
    },
    'task_merge_config': {
        'representation': 'pico',
        'merge_space': 'pico',
        'merge_method': 'ties',      # Match KnOTS & Core Space ('ties' instead of 'mean')
        'scaling_coeffs': 0.6,       # Match KnOTS & Core Space (0.6 instead of 1.0)
        'topK': 20,                  # Match KnOTS & Core Space (20)
        'lora_alpha': 16.0,
        'eps': 1e-12,
    },
    'eval_type': 'clip',
}
