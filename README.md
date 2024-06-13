## Exam Preparation Python

**1. Scalar, Vector, and Tensor difference:**
   - **Scalar:** Singular numerical entity. Examples include loss and accuracy.
   - **Vector:** Ordered collections of numerical values (scalars).
   - **Tensor:** Generalizes the concept of vectors and matrices. Vectors are 1-rank tensors, matrices are 2-rank tensors, and so on.

**2. Fashion MNIST and MNIST:**
   - **MNIST:** A dataset of handwritten digits.
   - **Fashion MNIST:** A dataset of clothing items.
   - Both datasets contain 60,000 training images and 10,000 testing images, with each image being in 28x28 grayscale format. MNIST originally had images in 128x128 format but converted to 28x28.

**3. OOP principles:**
   - Encapsulation, Inheritance, Polymorphism, Abstraction.

**4. Python Loops:**
   - `for`, `while`, `do while`.

**5. Compiler vs. Interpreter:**
   - Compiler translates the whole source code to machine language before the program runs.
   - Interpreter reads the code line by line as the code runs.

**6. Usage Cases of Sigmoid, Softmax, and ReLU:**
   - **Sigmoid:** Binary classification and logistic regression.
   - **Softmax:** Multiclass classification.
   - **ReLU:** Image recognition and computer vision.

**7. Python Modules:**
   - A single Python script containing functions and variables. The module name is the same as the file name without the `.py` suffix. E.g., file `isa.py` has the module name `isa`. Import using `import isa`.

**8. Class vs. Functions:**
   - Functions are reusable code blocks, while classes are blueprints for creating objects.

**9. Data Science and Data Analysis:**
   - Data analysis involves analyzing past data to inform present decisions. Data science combines data modeling and data collection.

**10. DL and ML:**
   - ML is a subset of AI. DL is a subset of ML. ML uses statistical algorithms and models to improve performance. DL involves neural networks and learns from large datasets.

**11. Mutable vs. Immutable:**
   - Immutable: String, int, tuple, frozenset.
   - Mutable: List, dictionary, set. Elements within frozenset and tuple can be mutable.

**12. CNN:**
   - Convolutional Neural Networks are deep learning algorithms used for image classification and object recognition tasks, utilizing three-dimensional data. Example: Face recognition.

**13. .py vs. .ipynb:**
   - `.py`: Regular Python file containing only code blocks.
   - `.ipynb`: Notebook file containing code blocks, execution results, and internal settings.

**14. Usage of Loss Function and Optimizer:**
   - Loss function shows the difference between predicted data and actual data.
   - Optimizer tries to minimize this difference.

**15. Tensorflow Operations:**
   ```python
   import tensorflow as tf
   a = tf.constant([3, 3, 3])
   b = tf.constant([2, 2, 2])  # Define tensors

   sum_result = tf.add(a, b)    # Addition
   diff_result = tf.subtract(a, b)    # Subtraction
   quot_result = tf.divide(a, b)    # Division
   prod_result = tf.multiply(a, b)    # Multiplication

   max, min, abs, log, and exp
   ```

**16. Set, Tuple, and List:**
   - Set: A list of unique elements.
   - Tuple: Immutable lists.
   - List: Mutable lists.

**17. Steps in Machine Learning:**
   - Gathering data
   - Pre-processing data (Normalization)
   - Training the model
   - Optimizing

**18. String int bilərsiz də yəqin aq**

**19. Pandas and Numpy:**
   - **Pandas:** Data manipulation and analysis. Missing data shown as `NaN`.
   - **Numpy:** Used for mathematical operations with arrays, tensors, etc.

**20. Shape Operations and Min-Max Operations:**
   - **Shape Operations:** Reshaping, Concatenation, Transposing, Splitting, Slicing.
   - **Min-Max Operations:** min, max, argmax, argmin, softmax.
