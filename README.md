# KNN Iris Classifier

A simple **K-Nearest Neighbors (KNN)** implementation in Java that classifies iris flowers into three species:
- *Iris Setosa*
- *Iris Versicolor*
- *Iris Virginica*

The project uses the classic **Iris dataset** for training and testing.

---

## How It Works
The algorithm calculates the Euclidean distance between data points and classifies each sample based on the **k nearest neighbors**.  
You can easily modify the value of `k` and test accuracy using different dataset splits.

---

## Installation Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/shehabeldin-mohamed/KNN-Iris-Classifier.git
   cd KNN-Iris-Classifier
2. **Move to src and compile the program**
   ```bash
   cd src
   javac KNN.java
3. **Run the program by specifying how many neighbours you want, training dataset, and test dataset.**
   **For example**
   ```bash
   java KNN.java 3 iris.data iris.test.data 
  
