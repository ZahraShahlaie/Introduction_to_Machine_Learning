البته! این هم متن کامل و بدون تغییر:

---

بله، حتماً! می‌توانم به شما در نوشتن کد پایتون برای پیاده‌سازی رگرسیون لجستیک کمک کنم. از آنجایی که فایل PDF مفاهیم نظری را پوشش می‌دهد، ما می‌توانیم این مفاهیم را به صورت عملی در پایتون کدنویسی کنیم.

برای شروع، بیایید گام‌های اصلی پیاده‌سازی رگرسیون لجستیک را در پایتون مرور کنیم و سپس برای هر گام، کد مربوطه را بنویسیم. ما یک پیاده‌سازی "از صفر" (from scratch) خواهیم داشت که به درک عمیق‌تر کمک می‌کند، و سپس می‌توانیم به سراغ استفاده از کتابخانه‌های آماده مانند Scikit-learn برویم.

گام‌های اصلی برای پیاده‌سازی رگرسیون لجستیک از صفر:

آماده‌سازی داده (Data Preparation):

ساخت یا بارگذاری یک مجموعه داده نمونه برای طبقه‌بندی دودویی.

افزودن بایاس ترم (Bias Term) به داده‌ها.

نرمال‌سازی ویژگی‌ها (Feature Scaling).

تعریف تابع سیگموید (Sigmoid Function).

تعریف تابع هزینه (Cost Function - Binary Cross-Entropy).

محاسبه گرادیان (Gradient Calculation).

پیاده‌سازی گرادیان دیسنت (Gradient Descent) برای بهینه‌سازی وزن‌ها.

تابع پیش‌بینی (Prediction Function).

ارزیابی مدل (Model Evaluation).

بیایید با گام اول شروع کنیم.

بخش ۱: آماده‌سازی داده‌ها و توابع پایه

۱.1. وارد کردن کتابخانه‌های لازم:
اولین قدم، وارد کردن کتابخانه‌های پایتون مورد نیاز است: numpy برای محاسبات عددی و matplotlib.pyplot برای رسم نمودارها.

Python

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

توضیح:

* numpy: برای کار با آرایه‌ها و انجام عملیات ریاضی (مثل ضرب ماتریسی، تابع نمایی exp و log).
* matplotlib.pyplot: برای بصری‌سازی داده‌ها و نتایج.
* sklearn.model\_selection.train\_test\_split: برای تقسیم داده‌ها به مجموعه‌های آموزش و تست.
* sklearn.preprocessing.StandardScaler: برای نرمال‌سازی ویژگی‌ها.
* sklearn.metrics: برای ارزیابی عملکرد مدل.

۱.2. ساخت یک مجموعه داده نمونه (Synthetic Dataset):
برای شروع، یک مجموعه داده ساده برای طبقه‌بندی دودویی ایجاد می‌کنیم.

Python

```python
# Generate a synthetic dataset for binary classification
np.random.seed(0) # for reproducibility

# Class 0 data
X_class0 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
y_class0 = np.zeros(100)

# Class 1 data
X_class1 = np.random.multivariate_normal([3, 3], [[1, 0.5], [0.5, 1]], 100)
y_class1 = np.ones(100)

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
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Synthetic Dataset for Binary Classification")
plt.legend()
plt.grid(True)
plt.show()
```

توضیح:

* np.random.multivariate\_normal: برای تولید داده‌هایی که از توزیع نرمال چندمتغیره پیروی می‌کنند. این به ما کمک می‌کند دو "ابر" از نقاط را برای دو کلاس ایجاد کنیم که تا حدی از هم جدا هستند.
* np.zeros و np.ones: برای ایجاد برچسب‌های 0 و 1.
* np.vstack و np.hstack: برای ترکیب داده‌های دو کلاس.
* np.random.permutation: برای درهم‌ریختن (shuffle) داده‌ها تا ترتیب کلاس‌ها روی مدل تأثیر نگذارد.
* نمودار plt.scatter: برای مشاهده چگونگی توزیع داده‌ها.

---

اگر نیاز داشتید می‌توانم ادامه گام‌های بعدی و کدهای مربوط به آنها را هم برایتان بنویسم.
۱.3. تقسیم داده به مجموعه‌های آموزش و تست و نرمال‌سازی (Scaling):
این گام برای ارزیابی صحیح عملکرد مدل بسیار مهم است.

```python
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Feature Scaling (Standardization)
# This helps gradient descent converge faster and perform better
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use transform, not fit_transform, on test set

# Add bias term (x0 = 1) to the features
# This allows us to incorporate the intercept term (w0) into the weight vector
X_train_final = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))
X_test_final = np.hstack((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled))

print("\nShape of X_train_final (with bias):", X_train_final.shape)
print("Shape of X_test_final (with bias):", X_test_final.shape)
```

توضیح:

* `train_test_split`: ۸۰٪ داده‌ها برای آموزش و ۲۰٪ برای تست استفاده می‌شوند. `random_state` برای اطمینان از تکرارپذیری تقسیم‌بندی است.
* `StandardScaler`: این ابزار ویژگی‌ها را طوری مقیاس‌بندی می‌کند که میانگین آن‌ها ۰ و واریانس آن‌ها ۱ شود. این کار باعث می‌شود گرادیان دیسنت سریع‌تر همگرا شود، زیرا ویژگی‌ها در یک بازه‌ی مشابه قرار می‌گیرند. مهم: `fit_transform` را فقط روی داده‌های آموزش استفاده کنید تا مدل مقیاس‌بندی را از داده‌های آموزش یاد بگیرد. سپس از `transform` برای اعمال همان مقیاس‌بندی به داده‌های تست استفاده کنید.
* اضافه کردن بایاس ترم (Bias Term): این گام بسیار مهم است. ما یک ستون از یک‌ها را به ابتدای ماتریس ویژگی‌ها اضافه می‌کنیم. این کار باعث می‌شود که عبارت بایاس (`w_0`) در مدل ما، به عنوان یک وزن جداگانه در بردار وزن کلی (`w`) قرار گیرد و نیازی نباشد به صورت جداگانه مدیریت شود. این شبیه به اضافه کردن `x_0=1` به بردار ویژگی‌ها است تا `w_0` به صورت `w_0 x_0` در ضرب `w^T x` قرار گیرد.

---

بخش ۲: تعریف توابع اصلی رگرسیون لجستیک (Sigmoid, Cost, Gradient)
حالا که داده‌ها آماده‌اند، می‌توانیم توابع کلیدی رگرسیون لجستیک را تعریف کنیم.

۲.۱. تابع سیگموید:
همانطور که در PDF توضیح داده شد،

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

```python
def sigmoid(z):
    """
    Computes the sigmoid function for the input z.
    Args:
        z (numpy.array or float): The input value(s) to the sigmoid function.
    Returns:
        numpy.array or float: The output of the sigmoid function, between 0 and 1.
    """
    return 1 / (1 + np.exp(-z))
```

توضیح:

* `np.exp(-z)`: محاسبه $e^{-z}$.
* این تابع برای هر مقدار $z$ (می‌تواند یک عدد یا یک آرایه numpy باشد) خروجی سیگموید را تولید می‌کند.
۲.۲. تابع هزینه (Binary Cross-Entropy Loss):
فرمول تابع هزینه در PDF به صورت
$J(w) = -\sum_{i=1}^n \left[ y^{(i)} \log\big(\sigma(w^T x^{(i)})\big) + (1 - y^{(i)}) \log\big(1 - \sigma(w^T x^{(i)})\big) \right]$
آمده است.

```python
def cost_function(X, y, w):
    """
    Computes the cost (Binary Cross-Entropy Loss) for logistic regression.
    Args:
        X (numpy.array): The feature matrix (with bias term).
        y (numpy.array): The true labels.
        w (numpy.array): The weight vector.
    Returns:
        float: The computed cost.
    """
    m = X.shape[0] # Number of training examples
    h = sigmoid(X @ w) # Predicted probabilities (h_theta(x))

    # Avoid log(0) by clipping probabilities
    epsilon = 1e-10
    h = np.clip(h, epsilon, 1 - epsilon)

    cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost
```

توضیح:

* m: تعداد نمونه‌های آموزشی.
* $X @ w$: این یک ضرب ماتریسی است (@ عملگر ضرب ماتریسی در پایتون است) که $w^T x$ را برای تمام نمونه‌ها به صورت یکجا محاسبه می‌کند. نتیجه یک بردار از مقادیر $z$ است.
* sigmoid(X @ w): این همان $h_\theta(x)$ یا $\sigma(w^T x)$ است که احتمال پیش‌بینی شده را می‌دهد.
* epsilon: اضافه کردن یک مقدار epsilon کوچک (1e-10) برای جلوگیری از log(0) است که منجر به خطای نامحدود می‌شود. np.clip مقادیر را بین epsilon و 1 - epsilon نگه می‌دارد.
* np.sum: تمام ترم‌ها را جمع می‌کند. (-1/m) برای میانگین‌گیری و منفی کردن است (چون ما می‌خواهیم منفی لگاریتم درست‌نمایی را کمینه کنیم).

---

۲.۳. محاسبه گرادیان:
فرمول گرادیان در PDF به صورت

$$
\nabla_w J(w) = \sum_{i=1}^n \left(\sigma(w^T x^{(i)}) - y^{(i)}\right) x^{(i)}
$$

آمده است.

```python
def compute_gradient(X, y, w):
    """
    Computes the gradient of the cost function with respect to the weights.
    Args:
        X (numpy.array): The feature matrix (with bias term).
        y (numpy.array): The true labels.
        w (numpy.array): The weight vector.
    Returns:
        numpy.array: The gradient vector.
    """
    m = X.shape[0]
    h = sigmoid(X @ w) # Predicted probabilities

    gradient = (1/m) * (X.T @ (h - y))
    return gradient
```

توضیح:

* $h - y$: این همان عبارت "خطا" یا "تفاوت" است $\sigma(w^T x^{(i)}) - y^{(i)}$.
* $X^T @ (h - y)$: این یک ضرب ماتریسی است که به طور کارآمد جمع $\sum_{i=1}^n \left(\sigma(w^T x^{(i)}) - y^{(i)}\right) x^{(i)}$ را برای تمام ویژگی‌ها (و بایاس) محاسبه می‌کند. $X^T$ ترانهاده (transpose) ماتریس ویژگی‌ها است.
* $1/m$: برای میانگین‌گیری گرادیان روی تمام نمونه‌ها.

---

بخش ۳: پیاده‌سازی گرادیان دیسنت و پیش‌بینی
حالا که توابع اصلی آماده‌اند، می‌توانیم الگوریتم گرادیان دیسنت را پیاده‌سازی کنیم تا وزن‌های بهینه را پیدا کنیم.

---

۳.۱. گرادیان دیسنت:
قاعده‌ی به‌روزرسانی وزن‌ها:

$$
w^{t+1} = w^t - \eta \nabla_w J(w^t)
$$

```python
def gradient_descent(X, y, w_initial, learning_rate, num_iterations):
    """
    Performs gradient descent to optimize the weight vector.
    Args:
        X (numpy.array): The feature matrix.
        y (numpy.array): The true labels.
        w_initial (numpy.array): The initial weight vector.
        learning_rate (float): The learning rate (eta).
        num_iterations (int): The number of iterations to run gradient descent.
    Returns:
        tuple: A tuple containing:
            - w (numpy.array): The optimized weight vector.
            - cost_history (list): A list of cost values at each iteration.
    """
    w = w_initial
    cost_history = []

    for i in range(num_iterations):
        grad = compute_gradient(X, y, w)
        w = w - learning_rate * grad # Update rule
        cost = cost_function(X, y, w)
        cost_history.append(cost)

        if i % 100 == 0: # Print cost every 100 iterations
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return w, cost_history
```

توضیح:

* $w_{initial}$: وزن‌ها را معمولاً با مقادیر کوچک تصادفی یا صفر مقداردهی اولیه می‌کنیم.
* حلقه for: گرادیان دیسنت را برای تعداد مشخصی `num_iterations` اجرا می‌کند.
* `cost_history`: برای ذخیره‌ی مقدار هزینه در هر تکرار، که می‌توانیم از آن برای بررسی همگرایی استفاده کنیم.


۳.۲. تابع پیش‌بینی:
بعد از آموزش مدل و یافتن وزن‌های بهینه، از این تابع برای پیش‌بینی کلاس نمونه‌های جدید استفاده می‌کنیم. قاعده‌ی تصمیم‌گیری: اگر
$h_\theta(x) \geq 0.5$، پیش‌بینی $y = 1$؛ در غیر این صورت $y = 0$.

```python
def predict(X, w):
    """
    Predicts the class labels (0 or 1) based on the learned weights.
    Args:
        X (numpy.array): The feature matrix (with bias term).
        w (numpy.array): The learned weight vector.
    Returns:
        numpy.array: An array of predicted class labels (0s and 1s).
    """
    probabilities = sigmoid(X @ w)
    return (probabilities >= 0.5).astype(int)
```

توضیح:

* `sigmoid(X @ w)`: احتمال پیش‌بینی شده برای هر نمونه را محاسبه می‌کند.
* `(probabilities >= 0.5)`: یک آرایه بولی (True/False) برمی‌گرداند.
* `.astype(int)`: مقادیر True را به 1 و False را به 0 تبدیل می‌کند.

---

بخش ۴: آموزش مدل و ارزیابی
حالا می‌توانیم تمام این قطعات کد را کنار هم بگذاریم و مدل رگرسیون لجستیک خود را آموزش دهیم و عملکرد آن را ارزیابی کنیم.

```python
# --- Model Training ---

# Initialize weights (number of features + 1 for bias)
num_features = X_train_final.shape[1]
w_initial = np.zeros(num_features)  # Can also use small random values

learning_rate = 0.1
num_iterations = 1000

print("Starting Gradient Descent...")
w_optimized, cost_history = gradient_descent(X_train_final, y_train, w_initial, learning_rate, num_iterations)
print("\nOptimized weights (w):", w_optimized)

# Plotting the cost history
plt.figure(figsize=(10, 6))
plt.plot(range(num_iterations), cost_history, color='purple')
plt.xlabel("Number of Iterations")
plt.ylabel("Cost (J(w))")
plt.title("Cost History during Gradient Descent")
plt.grid(True)
plt.show()

# --- Model Evaluation ---

# Predictions on training set
y_train_pred = predict(X_train_final, w_optimized)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print("Training Classification Report:\n", classification_report(y_train, y_train_pred))

# Predictions on test set
y_test_pred = predict(X_test_final, w_optimized)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")
print("Test Classification Report:\n", classification_report(y_test, y_test_pred))

# Confusion Matrix for test set
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix (Test Set):\n", cm)

# Plotting the decision boundary (only for 2 features + bias)
if num_features == 3:  # X_train_final has 3 columns (bias, feature1, feature2)
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0 (Actual)')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1 (Actual)')

    # Get the original (unscaled) feature ranges for plotting
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                           np.linspace(x2_min, x2_max, 100))

    # Apply the same scaler to the meshgrid points
    # Need to reshape for scaler.transform
    grid_points_scaled = scaler.transform(np.c_[xx1.ravel(), xx2.ravel()])
    grid_points_final = np.hstack((np.ones((grid_points_scaled.shape[0], 1)), grid_points_scaled))

    Z = predict(grid_points_final, w_optimized)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=plt.cm.RdBu)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.title('Logistic Regression Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()
```

توضیح ارزیابی:

* `accuracy_score`: درصد پیش‌بینی‌های صحیح را محاسبه می‌کند.
* `classification_report`: یک گزارش کامل از دقت (Precision)، فراخوانی (Recall) و امتیاز F1 برای هر کلاس و میانگین آن‌ها ارائه می‌دهد.
* `confusion_matrix`: ماتریس درهم‌ریختگی را نشان می‌دهد که تعداد پیش‌بینی‌های صحیح و نادرست برای هر کلاس را به تفکیک می‌دهد (True Positives, True Negatives, False Positives, False Negatives).
* رسم مرز تصمیم: برای داده‌های دو بعدی (با بایاس، یعنی در کل ۳ وزن)، می‌توانیم مرز تصمیم‌گیری را رسم کنیم. این خط نشان می‌دهد که مدل چگونه دو کلاس را از هم جدا کرده است. دقت کنید که نقاط meshgrid نیز باید با همان scaler که داده‌های آموزشی مقیاس‌بندی شدند، مقیاس‌بندی شوند تا مرز تصمیم در فضای اصلی نمایش داده شود.



این کد یک پیاده‌سازی کامل "از صفر" برای رگرسیون لجستیک است. اگر می‌خواهید با کتابخانه scikit-learn کار کنید، پیاده‌سازی بسیار ساده‌تر خواهد بود:

پیاده‌سازی رگرسیون لجستیک با Scikit-learn (جهت مقایسه)

```python
from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression model from scikit-learn
# solver='liblinear' is good for small datasets and supports L1/L2 regularization
# random_state for reproducibility
sklearn_model = LogisticRegression(solver='liblinear', random_state=42)

# Train the model using the scaled training data (without manually adding bias, sklearn handles it)
# Note: sklearn's LogisticRegression automatically adds the bias term internally
sklearn_model.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_test_pred_sklearn = sklearn_model.predict(X_test_scaled)

# Evaluate the scikit-learn model
print("\n--- Scikit-learn Logistic Regression Results ---")
print(f"Test Accuracy (Scikit-learn): {accuracy_score(y_test, y_test_pred_sklearn):.4f}")
print("Classification Report (Scikit-learn):\n", classification_report(y_test, y_test_pred_sklearn))
```

توضیح:
همانطور که می‌بینید، با scikit-learn، بخش‌های تعریف تابع هزینه، گرادیان و گرادیان دیسنت همگی در پشت صحنه مدیریت می‌شوند و کد بسیار خلاصه‌تر است. با این حال، پیاده‌سازی از صفر به درک عمیق‌تر اصول کار الگوریتم کمک می‌کند.

این کد به شما یک شروع عالی برای کار با رگرسیون لجستیک در پایتون می‌دهد. اگر سوالی در مورد هر بخش از کد یا نیاز به توضیح بیشتر دارید، بپرسید!
