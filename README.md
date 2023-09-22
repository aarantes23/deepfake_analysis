# Deepfake Detection and Image Classification Application

This Python application allows you to perform various tasks related to deepfake detection and image classification into two categories: "fake" and "real". The application includes the following functionalities:

## 1 - Downloading Images from "This Person Does Not Exist"

The [`download_fakes.ipynb`](download_fakes.ipynb) notebook downloads randomly generated images from the [This Person Does Not Exist](https://thispersondoesnotexist.com/) website and saves them to a local directory.

### Prerequisites

- [requests](https://pypi.org/project/requests/) library for making HTTP requests.

### Usage

1. Clone the repository or download the [`download_fakes.ipynb`](download_fakes.ipynb) notebook.
2. Make sure you have the `requests` library installed.
3. Execute the notebook.

**Expected Result:** The notebook should download a specified number of randomly generated images and save them to the local folder.

## 2 - Downloading Real Images from the CelebA Dataset

This part was not automated because it was blocked by Google Drive. So, access the CelebA dataset link available at [https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), download, and extract the images to the `img_align_celeba\real` folder.

## 3 - Downloading and Preparing Images from the LFW (Labeled Faces in the Wild) Dataset

**Optional step:** The [`download_reals.ipynb`](download_reals.ipynb) notebook downloads and prepares images from the LFW (Labeled Faces in the Wild) dataset, which is often used for facial recognition tasks. The goal of having these images is to use them to evaluate the model on a completely different dataset than it was trained on. If you choose not to download these images, you will need to obtain other images to apply the model.

### Prerequisites

- [scikit-learn](https://scikit-learn.org/stable/) library for downloading the dataset.
- [numpy](https://numpy.org/) and [pandas](https://pandas.pydata.org/) libraries for data manipulation.

### Usage

1. Clone the repository or download the [`download_reals.ipynb`](download_reals.ipynb) notebook.
2. Make sure you have the `scikit-learn`, `numpy`, and `pandas` libraries installed.
3. Execute the notebook.

**Expected Result:** The notebook should download the LFW dataset images and save them to the local folder.

## 4 - Training the Deepfake Detection Model with Fake and Real Images using TensorFlow and Keras

The [`tensor_flow.ipynb`](tensor_flow.ipynb) notebook trains a convolutional neural network (CNN) model to classify images into two categories: "fake" and "real". It uses randomly generated images and real images from the CelebA dataset.

### Prerequisites

- [tensorflow](https://www.tensorflow.org/), [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [Pillow (PIL)](https://pillow.readthedocs.io/en/stable/), and [scikit-learn](https://scikit-learn.org/stable/) libraries.

### Usage

1. Clone the repository or download the [`tensor_flow.ipynb`](tensor_flow.ipynb) notebook.
2. Make sure you have the required libraries installed and the images in the corresponding folders.
3. Execute the notebook.

**Expected Result:** The notebook should train a deepfake detection model using the available images and save the trained model.

## 5 - Classification of Fake and Real Images using a Pre-trained Model

The [`classification.ipynb`](classification.ipynb) notebook classifies a set of images as "fake" or "real" using a pre-trained model that was trained earlier. It uses images located in the 'validate' folder and compares the model's predictions with the expected labels defined in the `expected_labels` dictionary.

### Prerequisites

- [tensorflow](https://www.tensorflow.org/), [numpy](https://numpy.org/), [Pillow (PIL)](https://pillow.readthedocs.io/en/stable/), and [scikit-learn](https://scikit-learn.org/stable/) libraries.

### Usage

1. Clone the repository or download the [`classification.ipynb`](classification.ipynb) notebook.
2. Make sure you have the required libraries installed, the images in the 'validate' folder, and the expected labels defined in the `expected_labels` dictionary.
3. Execute the notebook.

**Expected Result:** The notebook should classify the images, print predictions, and count the number of real and fake images.

## Running in the Docker Environment

This entire application was developed and executed in a Docker environment using the image `docker pull jupyter/tensorflow-notebook:2023-09-18`. To run the notebooks, make sure you have the Docker environment configured and install all the dependencies mentioned in the prerequisites of each notebook.

## Buy Me a Coffee ☕

If you found this project useful and would like to support me, feel free to buy me a coffee and also connect on [LinkedIn](https://www.linkedin.com/in/aarantes23/)!

<a href="https://www.buymeacoffee.com/aarantes23" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

## License

This project is licensed under the [MIT License](LICENSE).

---

This application offers a variety of functionalities related to deepfake detection and image classification. Each notebook can be executed individually, depending on the tasks you want to perform. Be sure to follow the specific instructions in each notebook to make the most of the application.
