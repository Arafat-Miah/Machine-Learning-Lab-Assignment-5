# Machine Learning Lab - Assignment 5: Multi-class & Nonlinear Classification

This repository contains my solutions for **Assignment 5** of the Machine Learning course (521289S). The focus of this assignment is extending classification techniques to multi-class problems and implementing advanced nonlinear models using feature transformations and Gradient Descent with Automatic Differentiation (AD).

## 📂 Repository Structure

The solutions are organized into separate MATLAB functions and scripts as required by the assignment templates:

| File Name | Description |
| :--- | :--- |
| `task1_MultiClass_Classification.m` | Implementation of the **Fusion Rule** to classify samples based on the maximum score across multiple weight vectors. |
| `task2_One_Vs_All_Training.m` | Training a multi-class model by solving $C$ independent binary classification sub-problems with normalized weights. |
| `task3_MultiClass_Perceptron_Training.m` | Simultaneous optimization of all class weights using a single **Multi-class Perceptron** cost function. |
| `task4_MultiClass_Softmax_Training.m` | Implementation of the **Multi-class Softmax** (Cross-Entropy) cost function with the Log-Sum-Exp trick for numerical stability. |
| `task5_Nonlinear_Classification_Modeling.m` | Solving periodic classification (diagonal stripes) using a **Sinusoidal feature transformation**. |

---

## 📝 Task Details

### Task 1: Multi-class Classification (Fusion Rule)
[cite_start]Implemented the prediction logic for a model with $C > 2$ classes[cite: 5].
- [cite_start]**Theory**: For any data point $x$, the predicted label $y'$ is determined by picking the class $c$ that yields the largest value for the decision rule: $y' = \text{argmax}_{c=0, \dots, C-1} \hat{x}^T w_c$[cite: 61, 279].
- [cite_start]**Implementation**: Augmented the input with a bias term and adjusted MATLAB's 1-based indexing to match the 0-based class labels ($0, 1, 2$) used in the course[cite: 9, 22].

### Task 2: One-versus-All (OvA) Training
[cite_start]Implemented the OvA training strategy to decompose a multi-class problem into binary sub-problems[cite: 7].
- [cite_start]**Process**: For each class $c$, a binary Perceptron is trained where class $c$ is assigned label $+1$ and all other classes are assigned $-1$[cite: 17, 18].
- [cite_start]**Normalization**: Each weight vector $w_c$ is normalized by its Euclidean norm ($w_j \leftarrow \frac{w_j}{\|w_j\|_2}$) to allow for a fair comparison of scores during fusion[cite: 72, 74].

### Task 3: Multi-class Perceptron Training
[cite_start]Transitioned from independent training to joint optimization of all weights[cite: 115].
- [cite_start]**Cost Function**: Implemented the total cost function: $g(W) = \frac{1}{P} \sum_{p=1}^{P} [\max_{j=0, \dots, C-1}(\hat{x}_p^T w_j) - \hat{x}_p^T w_{y_p}]$[cite: 132].
- [cite_start]**Mechanism**: If the correct class does not have the highest score, the optimizer updates all decision boundaries simultaneously to reduce the error[cite: 130].

### Task 4: Multi-class Softmax Training
[cite_start]Implemented a probabilistic approach to multi-class classification[cite: 173].
- [cite_start]**Theory**: Used the Cross-Entropy cost function: $g(W) = \frac{1}{P} \sum_{p=1}^{P} [\log(\sum_{j=0}^{C-1} e^{\hat{x}_p^T w_j}) - \hat{x}_p^T w_{y_p}]$[cite: 194].
- [cite_start]**Optimization**: The output can be interpreted as a discrete probability distribution, providing a confidence value for each prediction[cite: 173, 174].

### Task 5: Nonlinear Two-class Modeling
[cite_start]Addressed a complex "diagonal stripes" dataset that is not linearly separable in its original space[cite: 196].
- [cite_start]**Feature Transformation**: Mapped original features into a periodic space using a Sine transformation: $f(x) = \sin(v_1 x_1 + v_2 x_2)$[cite: 197].
- **Joint Optimization**: Optimized the parameter vector $\Theta$, containing both the linear weights ($w$) and the internal transformation parameters ($v$), to achieve 100% classification accuracy.

---

## ⚠️ Repository Purpose & Academic Integrity

This repository is created solely to demonstrate the knowledge and practical skills I gained in machine learning optimization during this course.

**The code is:**

* ❌ **Not intended for reuse, redistribution, or submission by others**
* ❌ **Not shared for the purpose of passing coursework or assessments**
* ✅ **Maintained as a personal academic and technical portfolio artifact**

Any use of this material should respect academic integrity policies and course regulations.
