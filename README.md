# Pneumonia Detection

This repository contains a project for detecting pneumonia from chest X-ray images using a deep learning model. The dataset consists of labeled JPEG images (Pneumonia/Normal) from a retrospective cohort of pediatric patients aged 1-5 years from Guangzhou Women and Children Medical Center in Guangzhou.

## Author

**Patryk Klytta**

## Project Structure

The repository is organized as follows:

- `chest_xray/`: Directory containing the X-ray images categorized into `train`, `val`, and `test` subdirectories.
- `train/`: Training dataset.
- `val/`: Validation dataset.
- `test/`: Test dataset.
- `pneumonia_detection.ipynb`: Jupyter notebook containing the code for training and evaluating the model.

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

## Data Preprocessing

The images are preprocessed using the `ImageDataGenerator` class from Keras, which includes the following transformations:

- Rescaling
- Shearing
- Zooming
- Horizontal and vertical flipping
- Brightness adjustments
- Width shifting
- Rotation

## Model Architecture

The model is built using the ResNet50V2 architecture with the following modifications:

- Global average pooling layer
- Dense layer with ReLU activation
- Output layer with sigmoid activation for binary classification

The ResNet50V2 layers are frozen to leverage transfer learning, only training the added layers.

## Training

The model is trained using the following configurations:

- Optimizer: Adam
- Loss function: Binary Cross-Entropy
- Metrics: Accuracy
- Number of epochs: 30
- Batch size: 32

## Evaluation

The model's performance is evaluated on the test set, with metrics including accuracy, precision, recall, and F1-score. Additionally, confusion matrices are generated for a detailed performance analysis.

### Example Results

- **Training Accuracy**: 95.91%
- **Validation Accuracy**: 91.66%
- **Test Accuracy**: 91%

## Visualizations

The training and validation loss and accuracy are plotted to visualize the model's learning process. Confusion matrices are also plotted for the training, validation, and test sets.

## Model Saving

The trained model is saved in the `resnet_pneumonia_model_PKlytta.keras` file for future inference.

## How to Run

To train and evaluate the model, run the provided Jupyter notebook `pneumonia_detection.ipynb` or execute the script in your Python environment.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](https://www.mit.edu/~amini/LICENSE.md).

art. 74 ust. 1 Ustawa o prawie autorskim i prawach pokrewnych, [Zakres ochrony programów komputerowych](https://lexlege.pl/ustawa-o-prawie-autorskim-i-prawach-pokrewnych/a
