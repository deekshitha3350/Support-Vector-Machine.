# Support-Vector-Machine.
# Mushroom Classification using SVM and PCA

## Overview
This project involves building a machine learning model to classify mushrooms as either edible or poisonous based on their features. The dataset contains various categorical features describing the physical attributes of mushrooms. The primary tasks include data preprocessing, visualization, feature encoding, dimensionality reduction using PCA, and classification using Support Vector Machines (SVM) with different kernel types.

---

## Steps in the Project

### 1. **Dataset**
- The dataset used in this project is `mushroom.csv`.
- It contains various features of mushrooms, including the class label (`edible` or `poisonous`).
- Each feature is categorical.

### 2. **Exploratory Data Analysis (EDA)**
- Checked for missing values in the dataset.
- Visualized the distribution of the class label and other features using count plots.
- Examined the relationships between features.

### 3. **Data Preprocessing**
- Encoded categorical features using `LabelEncoder` to convert them into numeric values.
- Split the data into training and testing sets (80% train, 20% test).
- Standardized the features using `StandardScaler`.

### 4. **Dimensionality Reduction**
- Applied Principal Component Analysis (PCA) to reduce the feature dimensions to 2 for visualization and simplified classification.

### 5. **Classification using SVM**
- Built a Support Vector Machine (SVM) classifier using different kernels:
  - Linear
  - Polynomial
  - Radial Basis Function (RBF)
- Evaluated the performance of the SVM model on the test set.

### 6. **Performance Metrics**
- Used the following metrics to evaluate the model:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
- Visualized the decision boundary of the SVM model with PCA-transformed features.

---

## Installation and Setup

### Requirements
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

### Steps to Run
1. Clone the repository:
   ```bash
   git clone <repository-link>
   ```
2. Navigate to the project directory:
   ```bash
   cd mushroom-classification
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python mushroom_classification.py
   ```

---

## Code Highlights

### PCA Visualization
Visualized the dataset after applying PCA, reducing it to 2 dimensions:
```python
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='coolwarm', alpha=0.7)
plt.title("PCA Visualization of Mushroom Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
```

### Decision Boundary Plot
Visualized the decision boundaries of the SVM model:
```python
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')
    plt.title("SVM Decision Boundary")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()
```

### Kernel Comparison
Tested different SVM kernels:
```python
kernels = ['linear', 'poly', 'rbf']
for kernel in kernels:
    print(f"Kernel: {kernel}")
    svm_model = SVC(kernel=kernel, C=1.0, random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report:\n {classification_report(y_test, y_pred)}")
```

---

## Results
- PCA allowed effective visualization of the dataset in 2D.
- The linear kernel provided the best performance among the tested kernels.
- Achieved high classification metrics, indicating a well-performing model.

---

## Visualizations
- Distribution of class labels and features.
- PCA visualization.
- SVM decision boundaries.

---

## Future Work
- Experiment with other classifiers (e.g., Random Forest, XGBoost).
- Perform hyperparameter tuning for SVM to optimize performance.
- Extend the analysis to handle missing data more effectively.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
