# Diabetic Retinopathy Detection using Deep Learning

## Overview
This project detects the severity level of Diabetic Retinopathy from retinal fundus images using a deep learning model.  
The model classifies retinal scans into different severity stages to assist in early diagnosis and screening.

---

## Classes

| Label | Class Name |
|------|-------------|
| 0 | No_DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferate_DR |

---

## Features
- Retinal image classification
- Deep learning-based prediction
- Severity stage detection
- Single image inference support
- Model evaluation pipeline



---

## Technologies Used
- Python
- TensorFlow / PyTorch
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## Training the Model

Run the following command to train the model:

```bash
python train_model.py
```

---

## Evaluating the Model

Run:

```bash
python evaluate_model.py
```

---

## Testing on a Single Image

Open `predict_single.py` and modify:

```python
predict_image("file_name.png")
```

Then run:

```bash
python predict_single.py
```

---

## Dataset
This project uses retinal fundus image datasets for Diabetic Retinopathy classification.

You can use datasets such as:
- https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data
---

## Model Performance

Example metrics:

| Metric | Score |
|--------|-------|
| Accuracy | 91% |
| F1 Score | 0.89 |


---

## Results

- Accuracy and loss graphs
 <img width="400" height="400" alt="accuracy_curve" src="https://github.com/user-attachments/assets/31708ddf-0708-42b6-9da8-e7da75356c58" />
 <img width="400" height="400" alt="loss_curve" src="https://github.com/user-attachments/assets/67400c67-a329-494f-93cd-fb68227fe3d3" />

- Confusion matrix
  <img width="800" height="550" alt="confusion_matrix" src="https://github.com/user-attachments/assets/69ec168c-545d-44d7-b8f1-53cb73015e3c" />

---

## Installation

Clone the repository:

```bash
git clone <https://github.com/SanchiJoshi/Diabetic-Retinopathy-Detection-using-CNN>
cd Diabetic-Retinopathy-Detection-using-CNN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---
