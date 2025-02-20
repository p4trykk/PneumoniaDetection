# Pneumonia Detection

This repository contains a project for detecting pneumonia from chest X-ray images using a deep learning model. The dataset consists of labeled JPEG images (Pneumonia/Normal) from a retrospective cohort of pediatric patients aged 1-5 years from Guangzhou Women and Children Medical Center in Guangzhou.

## Author

**Patryk Klytta**

## Project Structure

The repository is organized as follows:
```
Pneumonia-Detection/
│-- chest_xray/
│   │-- train/
│   │   │-- NORMAL/        # Normal chest X-rays
│   │   │-- PNEUMONIA/     # Pneumonia cases (both viral and bacterial)
│   │-- val/
│   │   │-- NORMAL/
│   │   │-- PNEUMONIA/
│   │-- test/
│   │   │-- NORMAL/
│   │   │-- PNEUMONIA/
- `pneumonia_detection.ipynb`: Jupyter notebook containing the code for training and evaluating the model.
```

## Setup

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/pneumonia_detection.git
    cd pneumonia_detection
    ```

2. Install the required dependencies (libraries and packages):

3. Ensure TensorFlow is configured to use the GPU if available.

## Dataset

The dataset used in this project is organized into three subsets:

- **Training Set**: Used to train the model.
- **Validation Set**: Used to tune the model hyperparameters.
- **Test Set**: Used to evaluate the model's performance on unseen data.

Each dataset (train, val, test) contains two main categories:
- NORMAL/ – X-ray images of healthy lungs.
- PNEUMONIA/ – X-ray images showing pneumonia, further divided into:
- Bacterial Pneumonia – Identified with "bacteria" in the filename.
- Viral Pneumonia – Identified with "virus" in the filename.
To properly utilize the dataset, the code must extract this information from the filenames and label the images accordingly during preprocessing.

## Data Preprocessing

The images are preprocessed using the `ImageDataGenerator` class from Keras, which includes the following transformations:

- Rescaling
- Shearing
- Zooming
- Horizontal and vertical flipping
- Brightness adjustments
- Width shifting
- Rotation

<p align="center"><img src='https://github.com/p4trykk/PneumoniaDetection/blob/main/results/class_distribution1102.png'></p>


## Model Architecture

### Two classes model using ResNet50
The model is built using the ResNet50V2 architecture with the following modifications:

- Global average pooling layer
- Dense layer with ReLU activation
- Output layer with sigmoid activation for binary classification

The ResNet50V2 layers are frozen to leverage transfer learning, only training the added layers.

### 3 classes model using EfficientNetB3 - training pipeline
The model is implemented using EfficientNetB3 with additional regularization and a focal loss function to handle class imbalance. Below are the key steps:
1. Data Loading and Preprocessing
   - Images are loaded and resized to 300x300 pixels
   - Labels are assigned based on filenames
   - Class weights are computed to address dataset imbalance
   - Augmentations such as horizontal flip, rotation, and brightness contrast are applied
2. Model Definition
   - Uses EfficientNetB3 as the base model with additional dense layers
   - Dropout and L2 regularization are applied
   - The output layer has 3 neurons for multi-class classification (Normal, Bacterial, Viral)
3. Training Configuration
    -  Optimized with AdamW optimizer
    -  Uses focal loss to counter class imbalance
    -  Cosine annealing learning rate scheduler is implemented
    -  Model training includes early stopping and best model checkpoint saving
4. Evaluation
   - The trained model is evaluated using a test set
   - A confusion matrix is generated to visualize classification performance
   - A classification report is printed with precision, recall, and F1 scores

### Results
For EfficientNetB3 model (classify 3 classes):
<p align="center"><img src='https://github.com/p4trykk/PneumoniaDetection/blob/main/results/confusion_matrixEN3.png'></p>

For Resnet50 model:
<p align="center"><img src='https://github.com/p4trykk/PneumoniaDetection/blob/main/results/confusion_matrix_VGG16.png'></p>


## Model Saving

The trained model is saved in the `resnet_pneumonia_model_PKlytta.keras` file for future inference.

## How to Run

To train and evaluate the model, run the provided Jupyter notebook `pneumonia_detection.ipynb` or execute the script in your Python environment.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](https://www.mit.edu/~amini/LICENSE.md).

art. 74 ust. 1 Ustawa o prawie autorskim i prawach pokrewnych, [Zakres ochrony programów komputerowych](https://lexlege.pl/ustawa-o-prawie-autorskim-i-prawach-pokrewnych/a
