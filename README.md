# Concrete Crack Classification — Final Year Project

Binary image classification of concrete surface cracks using deep learning, comparing a custom CNN trained from scratch against a fine-tuned EfficientNetB0 transfer learning model.

## Dataset

[Concrete Crack Images for Classification](https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification) — 40,000 images (227×227 RGB), collected from METU Campus Buildings, split evenly between crack (`Positive`) and no-crack (`Negative`) classes. Originally sourced from 458 high-resolution images (4032×3024) using the patch-extraction method of Zhang et al. (2016).

## Methodology

- **Split**: 70% train / 15% validation / 15% test, stratified by class.
- **Model 1 — Custom CNN**: 4 convolutional blocks (Conv2D → BatchNorm → MaxPooling), trained from scratch.
- **Model 2 — Transfer Learning**: EfficientNetB0 pretrained on ImageNet, trained in two phases — frozen-base head training, then fine-tuning the top ~30 layers at a low learning rate.
- **Augmentation**: random flips, rotation, brightness/contrast (train set only).
- **Explainability**: Grad-CAM visualizations to verify the models attend to crack regions.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Custom CNN | 1.00 | 1.00 | 1.00 | 1.00 | 1.0000 |
| EfficientNetB0 (fine-tuned) | 1.00 | 1.00 | 1.00 | 1.00 | 1.0000 |

Full metrics, confusion matrices, and Grad-CAM examples are in [`results/`](results/).

## Limitations

- The public dataset release does not retain filenames tracing patches back to their 458 source images, so a strict group-aware split (preventing near-duplicate patches from the same source photo appearing in both train and test) was not possible. This dataset is also known in prior literature to reach near-ceiling accuracy easily. Both factors likely contribute to the near-perfect scores above; results should be interpreted with this in mind, and generalization to genuinely external/unseen surfaces has not yet been tested.

## Project Structure

```
notebooks/   → main Jupyter notebook (data pipeline, training, evaluation)
results/     → exported figures and metrics
docs/        → presentation materials
```

## How to Run

1. Open `notebooks/concrete_crack_classification.ipynb` in Kaggle Notebooks or Google Colab.
2. On Kaggle: attach the dataset via "Add Input" and run all cells (GPU accelerator recommended).
3. On Colab: the notebook downloads the dataset automatically via `kagglehub`.

## Requirements

See [`requirements.txt`](requirements.txt).

## Author

[Your name] — Final Year Project, [Your Program/University]
