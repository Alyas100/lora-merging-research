# Core Space and Pico Evaluation

## Before evaluating

Everything needed for setup and reproduction is explained in:

```text
/root/script.ipynb
```

Before any evaluation, open and run the notebook cells to:

1. Download the GTSRB, MNIST, and SVHN LoRA adapters.
2. Download the CLIP ViT-B/32 classification heads.
3. Clone or update `core-space-merging`, `pico`, or other latest paper in this domain.
4. Create and activate the `knots_env` Conda environment.
5. Download the GTSRB, MNIST, and SVHN datasets.

Follow the notebook instructions and change any required paths or settings in the notebook cells before running the evaluation. 

Make sure to use this exact pipeline in `core-space-merging` paper so no need to create eval pipeline from scratch if you created your own algorithm/idea, you can plug your new model to this pipeline to run the eval.
