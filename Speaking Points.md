#### Week 1

#### Week 2
**The Formatting Layer (Logits to Probabilities)**

**The Architectural Problem:** At the end of our network, we output an array of raw, unbounded numbers called "Logits" (e.g., `[10.5, -2.1, 5.0]`). From a data structures perspective, this array is useless for making a clean decision. We need to build a formatting block that transforms this into a probability distribution (where all numbers are between 0.0 and 1.0, and they all sum up to exactly 1.0).

**The Algorithmic Logic & Your Role:**

- **The Interface:** Your friend will provide a mathematical function (called Softmax) to do this conversion. You will treat it as a black box: `Input: Array of floats -> Output: Array of formatted floats`.
    
- **🚨 The Engineering Gotcha (Floating-Point Overflow):** This is where you step in. The math your friend uses relies on explosive exponential growth. If our network outputs a slightly large number (like `1000`), the computer's memory cannot hold the result. It triggers an `OverflowError`, turns the data into `NaN` (Not a Number), and the entire program crashes.
    
- **The Algorithmic Fix:** You need to write a "safety wrapper" around your friend's math. Before their math runs, your code will find the highest number in the array and subtract it from every other number.
    - _Tell your friend:_ "Mathematically, this subtraction changes nothing about your final result. But algorithmically, it guarantees the numbers stay small enough that the computer's memory won't overflow."

**The Evaluation API (The Loss Function)**

**The Architectural Problem:** Now we have an array of formatted predictions, and we have an array representing the true answer. We need to compare them and output a **single float** that represents the "badness" of our guess. This is the System State.

**The Algorithmic Logic & Your Role:**

- **The Interface:** Again, your friend will write the math function for this (likely called Mean Squared Error or Cross-Entropy). Your architecture will call this function: `Input: (Prediction Array, Truth Array) -> Output: Single Float`.
- **🚨 The Engineering Gotcha (The Divide/Log by Zero Crash):** If the network is extremely confident but totally wrong (e.g., it predicts `0.0` for the right answer), your friend's math will try to process a zero in a way that is mathematically impossible. The program will return `-infinity` or `NaN`, and crash the loop.
- **The Algorithmic Fix (Data Clipping):** You need to write a data-sanitization step before the math happens. You will write code that "clamps" the prediction array so that absolute zeros or absolute ones are impossible. You force any `0.0` to become `0.0000001`.
    - _Tell your friend:_ "I am going to slightly mutate the data before I feed it to your loss function to prevent a system crash. The change is so microscopic it won't affect your calculus, but it will keep our code running."

**Internal State vs. User Interface (Loss vs. Accuracy)**

**The Architectural Problem:** We need to track how well the network is doing over time, but the computer and the human user need two completely different types of information.

**The Algorithmic Logic & Your Role:** You need to build two separate tracking pipelines in your code:

1. **The Internal State (Loss):** * **Data Type:** A highly sensitive, continuous floating-point number.
    - **Purpose:** This is the metric the algorithm actually uses to navigate. It detects microscopic improvements. It is purely for the backend system.
2. **The User Interface (Accuracy):** * **Data Type:** A rigid Boolean logic check (True/False) converted into a percentage.
    - **Purpose:** This is for the console output. You will write a simple `if prediction == truth: count++` logic block. The algorithm cannot use this to learn because it's too blunt (it doesn't care if the network was 51% confident or 99% confident, a win is a win). But it's what humans want to read on the screen.

**1. Softmax Safety Wrapper**
Prevents memory overflow

```python
import numpy as np

def safe_probability_formatter(raw_logits, friends_softmax_math):
    """
    Prevents floating-point overflow before running the math.
    """
    # 1. Find the largest number in the raw output array
    max_value = np.max(raw_logits)
    
    # 2. Subtract it from everything (shifts data down so the max is exactly 0)
    # This mathematically changes nothing for Softmax, but saves the computer's memory.
    safe_logits = raw_logits - max_value
    
    # 3. Now that the data is memory-safe, pass it to your friend's math black-box
    final_probabilities = friends_softmax_math(safe_logits)
    
    return final_probabilities
```

**2. Loss Function Safety Wrapper**
Prevent the `log(0)` crash 

```python
def safe_loss_calculator(predictions, truth_labels, friends_loss_math):
    """
    Prevents divide-by-zero or log(0) crashes during loss calculation.
    """
    # 1. Define 'epsilon' - a microscopically small number
    epsilon = 1e-7
    
    # 2. "Clamp" the data. 
    # Any number lower than epsilon becomes epsilon. 
    # Any number higher than 1 - epsilon becomes 1 - epsilon.
    # This means absolute 0.0 and absolute 1.0 no longer exist in our array.
    safe_predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    
    # 3. Pass the crash-proof data into your friend's loss math
    system_loss = friends_loss_math(safe_predictions, truth_labels)
    
    return system_loss
```

**3. Loss vs Accuracy**
Internal state vs user interface

```python
def evaluate_network_performance(predictions, truth_labels, friends_loss_math):
    """
    Routes data to calculate both the Machine Metric (Loss) and Human Metric (Accuracy).
    """
    # --- 1. THE MACHINE METRIC (Internal State) ---
    # We use the safety wrapper from above to get the highly precise float
    # The algorithm will use this number in Week 3 to navigate and learn.
    loss_value = safe_loss_calculator(predictions, truth_labels, friends_loss_math)
    
    # --- 2. THE HUMAN METRIC (User Interface) ---
    # We don't need calculus here, just rigid boolean logic.
    # 'argmax' just finds the index of the highest probability.
    network_guess_index = np.argmax(predictions)
    actual_truth_index = np.argmax(truth_labels)
    
    # Did the network guess right? True or False.
    if network_guess_index == actual_truth_index:
        is_accurate = True 
    else:
        is_accurate = False

    # Return both so the system can learn, and we can read the console!
    return loss_value, is_accurate
```
#### 1. The Formatting Layer (Logits to Probabilities)

**The Architectural Problem:** The network outputs a 2D matrix of raw numbers called "Logits". If we have a batch of 32 images and 10 possible digit classes, our array shape is `(32, 10)`. These numbers are unbounded wildcards (e.g., `105.2` or `-8.4`). We must format this into a valid probability distribution (0.0 to 1.0, summing to 1.0 per row).

**The Algorithmic Logic:**
- **Row-by-Row Processing (`axis=1`):** We process the batch image by image. In NumPy, this means applying our logic across `axis=1` (the rows) while keeping the matrix dimensions intact.
- **The Overflow Crash:** The math relies on exponential growth (`e^x`). If the network outputs a large logit, calculating the exponent will exceed the computer's memory, returning `NaN` (Not a Number) and destroying the network.
- **The Engineering Fix:** Before running the exponent, we find the maximum value in *each row* and subtract it from that row. This safely shifts all numbers down. Mathematically, the final probabilities remain identical, but our system is now memory-safe.
- **The Final Normalization:** We sum the safe exponents per row and divide the row by that sum, forcing the array to total exactly 1.0.

**Code Implementation:**
```python
import numpy as np

# 1. Find max per image (axis=1) to prevent overflow
max_logits = np.max(logits, axis=1, keepdims=True) 

# 2. Shift numbers down (Safety Wrapper)
safe_logits = logits - max_logits                    

# 3. Exaggerate differences and normalize
exponentials = np.exp(safe_logits)                   
probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
```

#### 2. The Evaluation API (The Loss Function)

**The Architectural Problem:** We now have a `(32, 10)` array of probabilities. We need to compare it to the True Answers (which are One-Hot Encoded arrays, like `[0, 0, 1, 0...]`). We need an algorithm to measure the "distance" between our guess and the truth, outputting a single float: the Loss.

**The Algorithmic Logic:**
- **The "Masking" Concept:** Cross-Entropy is essentially a giant matrix filter. Since the Truth array is full of `0`s and has only one `1` per row, multiplying the Truth array by our Probability array zeroes out every prediction _except_ the correct one. The algorithm only evaluates the probability assigned to the correct class.
- **The Log(0) Crash:** The math uses logarithms. If our network is totally wrong and assigns a `0.0` probability to the correct class, the computer attempts to calculate `log(0)`. This returns `-infinity` and crashes the training loop.
- **The Engineering Fix:** We "clip" the probability array. We force any absolute `0.0` to become `1e-7` (a microscopic decimal).
- **The Batch Average:** After the math processes the masked arrays, we take the mean average across all 32 images to get a single float representing the system's error.

**Code Implementation:**

```python
# 1. Prevent log(0) system crash
safe_probs = np.clip(probabilities, 1e-7, 1.0 - 1e-7) 

# 2. Apply the Mask (Zero-out the wrong classes)
masked_log_probs = truth_labels * np.log(safe_probs)  

# 3. Sum the row (only the correct class remains) and average the batch
loss_per_image = np.sum(masked_log_probs, axis=1)     
system_loss = -np.mean(loss_per_image)                
```

#### 3. Internal State vs. User Interface (Loss vs. Accuracy)

**The Architectural Problem:** The system requires two separate metrics. The algorithm needs a highly precise number to navigate, but humans need a simple, readable percentage to understand if the network is actually working.

![[Pasted image 20260306012008.png|500]]

**The Algorithmic Logic:**

- **Loss is Continuous (For the Machine):** Loss is a highly sensitive `float64`. It acts as the algorithm's altimeter. If the Loss drops from `2.4501` to `2.4500`, the algorithm knows it is moving in the right direction, even if the final guess was still technically wrong. It provides the "slope" for learning.
- **Accuracy is Stepped (For the Human):** Humans want boolean logic: "Did it get it right?" We use `argmax` to find the index of the highest probability and compare it to the true index.
- **The Disconnect:** The machine cannot learn from Accuracy. Boolean logic (True/False) has no slope or gradient. It cannot tell the machine it is _getting closer_. Therefore, we must calculate both pipelines side-by-side.

**Code Implementation:**

```python
# 1. Find the index of the highest guess vs actual truth
predicted_indices = np.argmax(probabilities, axis=1) 
true_indices = np.argmax(truth_labels, axis=1)       

# 2. Boolean check (Creates an array of True/False)
correct_guesses = (predicted_indices == true_indices)

# 3. Calculate percentage (NumPy treats True as 1, False as 0)
human_accuracy = np.mean(correct_guesses) 
```