# Astrophysical Objects Image Dataset

## Overview
This dataset contains a collection of astrophysical object images organized for training, validating, and testing deep learning models — specifically designed to support Convolutional Neural Network (CNN) training for object classification tasks in astronomy and astrophysics.

The dataset was built by collecting and unifying images from multiple public sources and existing datasets on Kaggle and other open repositories. All images have been organized into structured class folders for consistency and ease of use in supervised learning.

---

## Dataset Structure
The dataset is divided into three main directories:
- train/
- validation
- test


Each of these directories contains subfolders corresponding to **12 astrophysical object classes**:

- `asteroid`  
- `black_hole`  
- `earth`  
- `galaxy`  
- `jupiter`  
- `mars`  
- `mercury`  
- `neptune`  
- `pluto`  
- `saturn`  
- `uranus`  
- `venus`

**Example structure:**
train/
├── asteroid/
├── black_hole/
├── earth/
├── galaxy/
├── ...
validation/
├── asteroid/
├── black_hole/
├── ...
test/
├── asteroid/
├── black_hole/
├── ...

---

## Purpose
The main goal of this dataset is to provide a **clean, structured, and ready-to-use dataset** for:
- Image classification tasks in astrophysics  
- Training Convolutional Neural Networks (CNNs)  
- Testing and validating computer vision models  
- Benchmarking performance on multi-class classification problems

This dataset is particularly suitable for educational projects, research, and experimentation with deep learning frameworks such as TensorFlow, PyTorch, or similar.

---

## Image Source & Preprocessing
- Images have been gathered from multiple public datasets and open-source repositories.
- Classes were curated to ensure **coherence** and **relevance** to their corresponding astrophysical objects.
- Basic filtering and cleaning were applied to remove duplicates or inconsistent images.

> Note: The dataset may contain images of different resolutions. It is recommended to apply preprocessing steps (e.g., resizing, normalization) according to your model requirements.

---

## Suggested Use
Typical use cases include:
- Training a CNN for image classification
- Transfer learning experiments
- Testing different image augmentation techniques
- Model benchmarking and comparison

Example in Python:
```python
import tensorflow as tf

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "path/to/train",
    image_size=(224, 224),
    batch_size=32
)

```

## Citation & License

This dataset was compiled from publicly available image datasets. If you use it in your work, please consider citing this Kaggle dataset page.


## Contributions

Contributions and improvements (e.g., cleaning, balancing, labeling enhancements) are welcome.
Feel free to fork, modify, and share your trained models or notebooks with the community.

## Acknowledgements

Special thanks to the authors and contributors of the original datasets on Kaggle and other open repositories that made this compilation possible.

