# Waste Material Classification with a CNN

A convolutional neural network that sorts photos of waste into 7 material types: Cardboard, Food Waste, Glass, Metal, Other, Paper and Plastic. The goal is to support automated sorting, so more waste gets recycled instead of going to landfill.

Built with TensorFlow / Keras and trained from scratch on a Colab T4 GPU.

## Results

- **Test accuracy:** 65.7%
- **Macro F1:** 0.65
- **Dataset:** 7,625 images, 7 imbalanced classes

## What I did

1. Loaded the images from class folders and checked data quality. All images were readable and 256×256. I found 985 duplicate images.
2. Handled class imbalance (Plastic has 4.25× more images than Cardboard) with a stratified 70/15/15 split and class weights.
3. Resized images to 128×128, scaled pixels to 0–1 and one-hot encoded the labels.
4. Compared 3 CNN configurations. Each has 3 blocks of Conv2D → BatchNorm → MaxPool, then Dense, Dropout and softmax. Training used Adam, categorical cross-entropy, early stopping and learning-rate reduction.
5. Evaluated the best model on a held-out test set with precision, recall, F1 and a confusion matrix.
6. Tested data augmentation (flip, rotation, zoom, contrast) against the same test set.

## Model comparison

| Config | Filters | Dropout | Learning rate | Val accuracy |
|---|---|---|---|---|
| A | 32-64-128 | 0.4 | 0.001 | 0.592 |
| **B (best)** | **64-128-256** | **0.5** | **0.001** | **0.668** |
| C | 32-64-128 | 0.3 | 0.0005 | 0.631 |

## Key findings

- **Best classes:** Cardboard (F1 0.78), Food Waste (0.73), Metal (0.70).
- **Hardest classes:** Paper (F1 0.51) and Other (0.55).
- **Most common mistakes:** Paper → Plastic, Glass → Plastic and Plastic → Other. These materials can look very similar in a photo.
- **Augmentation:** It lowered test accuracy to 52.7%. The model needed more training time to learn from the harder, changing images.
- The best model was still improving at the last epoch, so longer training should help.

## Next steps

- Try transfer learning with a pretrained model (MobileNetV2 or ResNet50).
- Remove duplicate images before splitting.
- Train longer, with gentler augmentation.

## How to run

1. Open `waste-classification-cnn.ipynb` in Google Colab (GPU runtime) or Jupyter.
2. Put the dataset ZIP (one folder per class) in the working folder, or set `DATASET_ZIP` to its path.
3. Run all cells.

The dataset is not included in this repo.

## Tech stack

Python, TensorFlow / Keras, scikit-learn, NumPy, pandas, Matplotlib, Seaborn
