باشه، متن ابتدایی‌ای که شما دادی رو بدون هیچ تغییری دقیقاً اینجا میارم:

---

که در فایل PDF دوم (02-Linear\_Classification.pdf) مطرح شد، کمک کنم.

با توجه به محتوای فایل دوم، تمرکز اصلی ما بر روی:

تابع تشخیص خطی (Linear Discriminant Function)

پرسپترون (Perceptron): به عنوان یک طبقه‌بندی‌کننده خطی

قاعده‌ی به‌روزرسانی پرسپترون (Perceptron Update Rule) (هم Batch و هم Single-sample)

محدودیت پرسپترون (فقط برای داده‌های جداسازی‌پذیر خطی)

الگوریتم Pocket (برای داده‌های غیرجداسازی‌پذیر خطی)

بیایید با یک پیاده‌سازی "از صفر" (from scratch) برای پرسپترون شروع کنیم، سپس به الگوریتم Pocket بپردازیم و در نهایت با استفاده از کتابخانه‌ی scikit-learn این موارد را پیاده‌سازی کنیم.

بخش ۱: آماده‌سازی داده‌ها و توابع پایه
۱.۱. وارد کردن کتابخانه‌های لازم:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
# For Pocket Algorithm example, we might need a slightly non-linearly separable dataset
from sklearn.datasets import make_classification
```

توضیح:

numpy: برای محاسبات عددی و آرایه‌ها.

matplotlib.pyplot: برای بصری‌سازی.

sklearn.model\_selection.train\_test\_split: برای تقسیم داده‌ها.

sklearn.preprocessing.StandardScaler: برای نرمال‌سازی ویژگی‌ها.

sklearn.metrics: برای ارزیابی عملکرد.

sklearn.datasets.make\_classification: برای ساخت مجموعه داده‌های طبقه‌بندی مصنوعی.

هینو در هروحی بیار

بدون تغییر

---

این متن کامل دقیقاً همان چیزی است که فرستادید، آماده‌ام ادامه بدم یا بخش بعدی رو براتون بنویسم؟


دقیقاً همان‌طور که خواستی، متن کامل و کدها را بدون تغییر اینجا آوردم:

---

۱.2. ساخت یک مجموعه داده نمونه (Synthetic Linearly Separable Dataset):
برای پرسپترون، به داده‌های جداسازی‌پذیر خطی نیاز داریم.

```python
# Generate a synthetic dataset that is linearly separable for Perceptron
np.random.seed(42) # for reproducibility

# Class 0 data
X_class0 = np.random.multivariate_normal([1, 1], [[0.5, 0], [0, 0.5]], 50)
y_class0 = -np.ones(50) # Use -1 and +1 for Perceptron as in the PDF 

# Class 1 data
X_class1 = np.random.multivariate_normal([3, 3], [[0.5, 0], [0, 0.5]], 50)
y_class1 = np.ones(50) # Use -1 and +1 for Perceptron as in the PDF 

X = np.vstack((X_class0, X_class1)) # Stack features vertically
y = np.hstack((y_class0, y_class1)) # Stack labels horizontally

# Shuffle the dataset
shuffle_idx = np.random.permutation(len(y))
X = X[shuffle_idx]
y = y[shuffle_idx]

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Visualize the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', marker='x', label='Class -1')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='o', label='Class +1')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Linearly Separable Synthetic Dataset")
plt.legend()
plt.grid(True)
plt.show()
```

توضیح:

* برای پرسپترون، طبق PDF ، اغلب از برچسب‌های -1 و +1 به جای 0 و 1 استفاده می‌شود.
* داده‌ها طوری تولید شده‌اند که به وضوح جداسازی‌پذیر خطی باشند.

---

۱.۳. تقسیم داده به مجموعه‌های آموزش و تست و نرمال‌سازی (Scaling) و افزودن بایاس:

```python
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Standardization) [cite: 145]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add bias term (x0 = 1) to the features. This is crucial for linear classifiers. 
X_train_final = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))
X_test_final = np.hstack((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled))

print("\nShape of X_train_final (with bias):", X_train_final.shape)
print("Shape of X_test_final (with bias):", X_test_final.shape)
```

توضیح:

* اضافه کردن بایاس ترم (یک ستون از 1ها) به ماتریس ویژگی‌ها، به مدل اجازه می‌دهد تا ترم بایاس (
  w\_0) را به عنوان بخشی از بردار وزن w یاد بگیرد.

---

بخش ۲: پیاده‌سازی الگوریتم پرسپترون (از صفر)
۲.۱. تابع پیش‌بینی (Decision Function) پرسپترون:

خروجی پرسپترون بر اساس علامت $w^T x + w_0$ است.

```python
def predict_perceptron(X, w):
    """
    Predicts class labels (+1 or -1) using the perceptron's linear decision function.
    Args:
        X (numpy.array): Feature matrix (with bias term).
        w (numpy.array): Weight vector.
    Returns:
        numpy.array: Predicted labels (+1 or -1).
    """
    # h(x) = w^T x + w_0 
    scores = X @ w
    # Decision Rule: C1 if score >= 0, else C2. Here, +1 if >=0, -1 otherwise. [cite: 137, 157]
    return np.where(scores >= 0, 1, -1)
```

توضیح:

* `np.where(scores >= 0, 1, -1)`: اگر امتیاز (scores) بزرگتر یا مساوی 0 باشد، 1 را برمی‌گرداند؛ در غیر این صورت -1 را.

---

اگر قسمت بعدی رو هم می‌خوای، آماده‌ام!


دقیقاً متن و کدهایی که دادی رو بدون تغییر میارم:

---

۲.2. الگوریتم Perceptron Batch (دسته ای):

```python
def perceptron_batch_train(X, y, learning_rate, num_epochs):
    """
    Trains a Perceptron model using the Batch Perceptron algorithm.
    Args:
        X (numpy.array): Training feature matrix (with bias term).
        y (numpy.array): True labels (-1 or +1).
        learning_rate (float): The learning rate (eta).
        num_epochs (int): Number of passes over the training data.
    Returns:
        tuple: (optimized_weights, error_history)
    """
    num_features = X.shape[1]
    w = np.zeros(num_features) # Initialize weights to zeros 
    error_history = []

    for epoch in range(num_epochs):
        misclassified_mask = (predict_perceptron(X, w) != y) # Find misclassified points
        misclassified_X = X[misclassified_mask]
        misclassified_y = y[misclassified_mask]

        if len(misclassified_y) == 0:
            print(f"Epoch {epoch}: All points correctly classified. Converged.")
            error_history.append(0)
            break # Converged for linearly separable data [cite: 192]

        # Perceptron Update Rule (Batch): w <- w + eta * sum(y_i * x_i for misclassified points) 
        # In the PDF, gradient is -sum(y_i * x_i), so update is w - eta * (-sum(...)) = w + eta * sum(...)
        update_term = np.sum(misclassified_y[:, np.newaxis] * misclassified_X, axis=0)
        w = w + learning_rate * update_term

        # Calculate number of errors for history
        num_errors = np.sum(misclassified_mask)
        error_history.append(num_errors / X.shape[0]) # Store classification error rate

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Number of errors = {num_errors}/{X.shape[0]}")

    return w, error_history
```

توضیح:

* num\_epochs: تعداد دفعاتی که بر روی کل مجموعه داده آموزشی (Batch) تکرار می‌شود.
* misclassified\_mask: یک آرایه بولی که نشان می‌دهد کدام نقاط اشتباه طبقه‌بندی شده‌اند.
* `np.sum(misclassified_y[:, np.newaxis] * misclassified_X, axis=0)`: این بخش گرادیان را جمع می‌کند (sum\_iinMy\_ix\_i) همانطور که در فرمول به‌روزرسانی پرسپترون آمده است (که علامت گرادیان در PDF منفی است، پس در به‌روزرسانی مثبت می‌شود).

---

۲.3. الگوریتم Perceptron Single-sample (تک‌نمونه - SGD):

```python
def perceptron_single_sample_train(X, y, learning_rate, num_epochs):
    """
    Trains a Perceptron model using the Single-sample Perceptron (SGD) algorithm.
    Args:
        X (numpy.array): Training feature matrix (with bias term).
        y (numpy.array): True labels (-1 or +1).
        learning_rate (float): The learning rate (eta).
        num_epochs (int): Number of passes (epochs) over the training data.
    Returns:
        tuple: (optimized_weights, error_history)
    """
    num_features = X.shape[1]
    w = np.zeros(num_features) # Initialize weights 
    error_history = []
    
    m = X.shape[0]

    for epoch in range(num_epochs):
        num_misclassified_in_epoch = 0
        # Shuffle data for better SGD performance
        indices = np.random.permutation(m)
        
        for i in indices:
            x_i = X[i]
            y_i = y[i]
            
            prediction = predict_perceptron(x_i[np.newaxis, :], w) # Predict for single sample
            
            if prediction != y_i: # If misclassified 
                w = w + learning_rate * y_i * x_i # Perceptron Update Rule (SGD) 
                num_misclassified_in_epoch += 1
        
        error_history.append(num_misclassified_in_epoch / m)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Number of errors = {num_misclassified_in_epoch}/{m}")
            if num_misclassified_in_epoch == 0:
                print(f"Epoch {epoch}: All points correctly classified. Converged.")
                break # Converged for linearly separable data [cite: 196]

    return w, error_history
```

توضیح:

* `np.random.permutation(m)`: داده‌ها را در هر اپوک (Epoch) به صورت تصادفی درهم می‌ریزد تا مدل به ترتیب خاصی از نمونه‌ها وابسته نباشد.
* به‌روزرسانی وزن‌ها پس از پردازش هر نمونه اشتباه طبقه‌بندی شده انجام می‌شود.

---

اگر متن یا کد بیشتری داری، بفرست دقیق برات کپی می‌کنم.


متن و کدهای کامل طبق درخواست شما، بدون هیچ تغییری اینجا آورده شده است:

---

**بخش ۳: آموزش و ارزیابی مدل پرسپترون**
حالا می‌توانیم پرسپترون را آموزش دهیم و عملکرد آن را ارزیابی کنیم.

```python
# --- Training Perceptron Batch ---
print("\n--- Training Perceptron (Batch) ---")
w_batch, error_history_batch = perceptron_batch_train(X_train_final, y_train, learning_rate=0.01, num_epochs=100)

# --- Training Perceptron Single-sample ---
print("\n--- Training Perceptron (Single-sample) ---")
w_sgd, error_history_sgd = perceptron_single_sample_train(X_train_final, y_train, learning_rate=0.01, num_epochs=100)

# --- Evaluation ---
print("\n--- Evaluation Perceptron (Batch) ---")
y_train_pred_batch = predict_perceptron(X_train_final, w_batch)
train_accuracy_batch = accuracy_score(y_train, y_train_pred_batch)
print(f"Training Accuracy (Batch): {train_accuracy_batch:.4f}")
y_test_pred_batch = predict_perceptron(X_test_final, w_batch)
test_accuracy_batch = accuracy_score(y_test, y_test_pred_batch)
print(f"Test Accuracy (Batch): {test_accuracy_batch:.4f}")
print("Test Classification Report (Batch):\n", classification_report(y_test, y_test_pred_batch))


print("\n--- Evaluation Perceptron (Single-sample) ---")
y_train_pred_sgd = predict_perceptron(X_train_final, w_sgd)
train_accuracy_sgd = accuracy_score(y_train, y_train_pred_sgd)
print(f"Training Accuracy (Single-sample): {train_accuracy_sgd:.4f}")
y_test_pred_sgd = predict_perceptron(X_test_final, w_sgd)
test_accuracy_sgd = accuracy_score(y_test, y_test_pred_sgd)
print(f"Test Accuracy (Single-sample): {test_accuracy_sgd:.4f}")
print("Test Classification Report (Single-sample):\n", classification_report(y_test, y_test_pred_sgd))

# Plotting error history for both
plt.figure(figsize=(12, 6))
plt.plot(range(len(error_history_batch)), error_history_batch, label='Batch Perceptron Error', color='red')
plt.plot(range(len(error_history_sgd)), error_history_sgd, label='Single-sample Perceptron Error', color='blue', linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Error Rate")
plt.title("Perceptron Error History")
plt.legend()
plt.grid(True)
plt.show()

# Plotting decision boundary for a 2D dataset
if X_train_final.shape[1] == 3: # If 2 features + bias term
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', marker='x', label='Class -1 (Actual)')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='o', label='Class +1 (Actual)')

    # Create meshgrid for decision boundary plotting
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Apply scaling and add bias to meshgrid points
    grid_points_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    grid_points_final = np.hstack((np.ones((grid_points_scaled.shape[0], 1)), grid_points_scaled))

    # Get predictions for the meshgrid
    Z = predict_perceptron(grid_points_final, w_batch) # Using batch perceptron weights for plot
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.title('Perceptron Decision Boundary (Batch Trained)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()
```

---

**بخش ۴: پیاده‌سازی الگوریتم Pocket (برای داده‌های غیر جداسازی‌پذیر خطی)**
الگوریتم پرسپترون برای داده‌های غیر جداسازی‌پذیر خطی همگرا نمی‌شود. الگوریتم Pocket این مشکل را با نگهداری بهترین وزن‌ها در طول فرآیند آموزش حل می‌کند.

---

۴.۱. ساخت یک مجموعه داده غیر جداسازی‌پذیر خطی:

```python
# Generate a synthetic dataset that is NOT linearly separable for Pocket Algorithm
np.random.seed(42)

# Make a more complex, non-linearly separable dataset
X_nl, y_nl = make_classification(n_samples=200, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, flip_y=0.1, random_state=42)
# Convert y to -1 and +1 for Perceptron
y_nl = np.where(y_nl == 0, -1, 1)

# Split and scale
X_nl_train, X_nl_test, y_nl_train, y_nl_test = train_test_split(X_nl, y_nl, test_size=0.2, random_state=42)

scaler_nl = StandardScaler()
X_nl_train_scaled = scaler_nl.fit_transform(X_nl_train)
X_nl_test_scaled = scaler_nl.transform(X_nl_test)

X_nl_train_final = np.hstack((np.ones((X_nl_train_scaled.shape[0], 1)), X_nl_train_scaled))
X_nl_test_final = np.hstack((np.ones((X_nl_test_scaled.shape[0], 1)), X_nl_test_scaled))

# Visualize the non-linearly separable dataset
plt.figure(figsize=(8, 6))
plt.scatter(X_nl[y_nl == -1][:, 0], X_nl[y_nl == -1][:, 1], color='red', marker='x', label='Class -1')
plt.scatter(X_nl[y_nl == 1][:, 0], X_nl[y_nl == 1][:, 1], color='blue', marker='o', label='Class +1')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Non-Linearly Separable Synthetic Dataset (for Pocket)")
plt.legend()
plt.grid(True)
plt.show()
```

---

هر زمان خواستی ادامه رو هم بیارم یا توضیح بدم، در خدمتم.


متن و کد کامل طبق درخواست شما، دقیق و بدون تغییر اینجا آورده شده است:

---

### ۴.۲. پیاده‌سازی الگوریتم Pocket:

```python
def pocket_perceptron_train(X, y, learning_rate, num_epochs):
    """
    Trains a Perceptron model using the Pocket Algorithm for non-linearly separable data.
    Args:
        X (numpy.array): Training feature matrix (with bias term).
        y (numpy.array): True labels (-1 or +1).
        learning_rate (float): The learning rate (eta).
        num_epochs (int): Number of passes (epochs) over the training data.
    Returns:
        tuple: (optimized_weights_best, error_history)
    """
    num_features = X.shape[1]
    w = np.zeros(num_features) # Current weights 
    w_best = np.copy(w) # Best weights found so far 

    # Calculate initial number of errors for w_best
    initial_predictions = predict_perceptron(X, w_best)
    min_errors = np.sum(initial_predictions != y)

    error_history = [min_errors / X.shape[0]] # Store initial error rate

    m = X.shape[0]

    for epoch in range(num_epochs):
        # Shuffle data for SGD-like updates
        indices = np.random.permutation(m)
        
        current_epoch_errors = 0
        for i in indices:
            x_i = X[i]
            y_i = y[i]
            
            prediction = predict_perceptron(x_i[np.newaxis, :], w)
            
            if prediction != y_i: # If misclassified 
                w_new = w + learning_rate * y_i * x_i # Update candidate weights 
                
                # Check if w_new is better than w_best 
                predictions_with_w_new = predict_perceptron(X, w_new)
                errors_with_w_new = np.sum(predictions_with_w_new != y)
                
                if errors_with_w_new < min_errors:  # [cite: 209]
                    w_best = np.copy(w_new)          # [cite: 209]
                    min_errors = errors_with_w_new  # [cite: 209]
                    w = np.copy(w_new)  # Update current w to the new best, or keep the old one if w_new isn't better than w_best. This is a common heuristic.
                else:
                    w = np.copy(w_new)  # Continue with the updated weight, even if not the best so far for the whole dataset [cite: 206]
                current_epoch_errors += 1
        
        # Store the best error rate found in this epoch for history
        error_history.append(min_errors / X.shape[0])

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Best errors so far = {min_errors}/{m}, Current epoch misclassifications = {current_epoch_errors}/{m}")

    return w_best, error_history
```

---

### ۴.۳. آموزش و ارزیابی الگوریتم Pocket:

```python
# --- Training Pocket Perceptron ---
print("\n--- Training Pocket Perceptron ---")
w_pocket, error_history_pocket = pocket_perceptron_train(X_nl_train_final, y_nl_train, learning_rate=0.01, num_epochs=500)

# --- Evaluation Pocket Perceptron ---
print("\n--- Evaluation Pocket Perceptron ---")
y_train_pred_pocket = predict_perceptron(X_nl_train_final, w_pocket)
train_accuracy_pocket = accuracy_score(y_nl_train, y_train_pred_pocket)
print(f"Training Accuracy (Pocket): {train_accuracy_pocket:.4f}")

y_test_pred_pocket = predict_perceptron(X_nl_test_final, w_pocket)
test_accuracy_pocket = accuracy_score(y_nl_test, y_test_pred_pocket)
print(f"Test Accuracy (Pocket): {test_accuracy_pocket:.4f}")
print("Test Classification Report (Pocket):\n", classification_report(y_nl_test, y_test_pred_pocket))

# Plotting error history for Pocket
plt.figure(figsize=(10, 6))
plt.plot(range(len(error_history_pocket)), error_history_pocket, label='Pocket Perceptron Best Error', color='green')
plt.xlabel("Epochs")
plt.ylabel("Best Error Rate So Far")
plt.title("Pocket Perceptron Error History")
plt.legend()
plt.grid(True)
plt.show()

# Plotting decision boundary for Pocket (2D dataset)
if X_nl_train_final.shape[1] == 3:  # If 2 features + bias term
    plt.figure(figsize=(8, 6))
    plt.scatter(X_nl[y_nl == -1][:, 0], X_nl[y_nl == -1][:, 1], color='red', marker='x', label='Class -1 (Actual)')
    plt.scatter(X_nl[y_nl == 1][:, 0], X_nl[y_nl == 1][:, 1], color='blue', marker='o', label='Class +1 (Actual)')

    x_min, x_max = X_nl[:, 0].min() - 0.5, X_nl[:, 0].max() + 0.5
    y_min, y_max = X_nl[:, 1].min() - 0.5, X_nl[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    grid_points_scaled = scaler_nl.transform(np.c_[xx.ravel(), yy.ravel()])
    grid_points_final = np.hstack((np.ones((grid_points_scaled.shape[0], 1)), grid_points_scaled))

    Z = predict_perceptron(grid_points_final, w_pocket)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.title('Pocket Perceptron Decision Boundary (Best Weights)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()
```

---

### بخش ۵: پیاده‌سازی طبقه‌بندی خطی با Scikit-learn (جهت مقایسه)

Scikit-learn ابزارهای قدرتمندی برای طبقه‌بندی خطی ارائه می‌دهد، از جمله Perceptron و SVM (Support Vector Machine) که در نهایت یک طبقه‌بندی‌کننده خطی است.

```python
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC  # Linear Support Vector Classifier

# --- Scikit-learn Perceptron ---
print("\n--- Scikit-learn Perceptron ---")
# Perceptron in sklearn uses SGD and typically converges for linearly separable data.
# max_iter is for number of epochs, tol is for stopping criterion
# fit_intercept=True by default to add bias term
sklearn_perceptron = Perceptron(random_state=42, max_iter=1000, tol=1e-3)
sklearn_perceptron.fit(X_train_scaled, y_train)  # Note: use X_train_scaled, sklearn adds bias internally

y_test_pred_sklearn_p = sklearn_perceptron.predict(X_test_scaled)
print(f"Test Accuracy (Sklearn Perceptron): {accuracy_score(y_test, y_test_pred_sklearn_p):.4f}")
print("Classification Report (Sklearn Perceptron):\n", classification_report(y_test, y_test_pred_sklearn_p))

# --- Scikit-learn LinearSVC (a robust Linear Classifier) ---
print("\n--- Scikit-learn LinearSVC ---")
# LinearSVC is a C-based implementation of linear SVM, good for large datasets.
# It can also handle non-linearly separable data by finding the best linear separation.
sklearn_linear_svc = LinearSVC(random_state=42, max_iter=10000, tol=1e-4)  # Increased max_iter for convergence
sklearn_linear_svc.fit(X_nl_train_scaled, y_nl_train)  # Using the non-linearly separable data

y_test_pred_sklearn_svc = sklearn_linear_svc.predict(X_nl_test_scaled)
print(f"Test Accuracy (Sklearn LinearSVC on NL data): {accuracy_score(y_nl_test, y_test_pred_sklearn_svc):.4f}")
print("Classification Report (Sklearn LinearSVC on NL data):\n", classification_report(y_nl_test, y_test_pred_sklearn_svc))
```

---

**توضیح:**

* `sklearn.linear_model.Perceptron`: پیاده‌سازی Perceptron در scikit-learn، که از SGD استفاده می‌کند و برای داده‌های جداسازی‌پذیر خطی مناسب است.
* `sklearn.svm.LinearSVC`: یک طبقه‌بندی‌کننده بردار پشتیبان خطی (Linear Support Vector Classifier) است. این مدل می‌تواند یک مرز خطی بهینه پیدا کند، حتی اگر داده‌ها کاملاً جداسازی‌پذیر خطی نباشند (با بهینه‌سازی حاشیه). این مدل قوی‌تر از پرسپترون ساده است و در عمل بیشتر استفاده می‌شود.

---

اگر به ادامه یا توضیح هر بخش نیاز داشتی بگو تا کامل‌تر توضیح بدم یا قسمت بعدی رو بیارم.
