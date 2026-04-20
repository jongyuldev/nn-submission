# Building a Convolutional Neural Network from Scratch

## Motivation

If you've used PyTorch or Keras to build image classifiers, you've likely written
`nn.Conv2d()` or `model.fit()` without fully understanding what happens underneath.
Building a CNN from scratch — using only NumPy and pure Python — forces you to
understand every layer, every formula, and every design decision. This makes you
a significantly better engineer when you return to libraries, because you know
*why* things work, not just how to use them.

CNNs underpin a huge range of real-world applications: medical image analysis,
object detection, pose estimation, and image segmentation. The model built
alongside this guide classifies dermoscopic skin lesion images across 7 classes —
a genuinely useful medical application that demonstrates what a well-implemented
scratch model can achieve.

---

## Background

### Who is this guide for?

This guide is for learners who are comfortable writing Python and have already used
a deep learning library (PyTorch, Keras, or TensorFlow) to build and train a basic
model. You should be familiar with concepts like training loops, loss functions,
and accuracy metrics at a surface level. You do **not** need prior experience
implementing these mathematically — that is the point of this exercise.

### What you need before starting

- **Python and NumPy** — almost all implementation uses array operations
- **Basic CNN concepts** — if you've used `nn.Conv2d` before, you're ready
- **Basic calculus** — you don't need to derive formulas yourself, but understanding
  that gradients represent rates of change will help considerably

If you need to build your baseline CNN knowledge using libraries first:

- [PyTorch CNN Tutorial — DataCamp](https://www.datacamp.com/tutorial/pytorch-cnn-tutorial):
  Walks through building a CNN for MNIST digit recognition using PyTorch. Good
  for establishing a foundation before going from scratch.

---

## Learning Materials

The resources below are ordered roughly by where they sit in the learning journey.

- [Building a Neural Network from Scratch — freeCodeCamp](https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/):
  A well-structured walkthrough for building a basic fully-connected neural network from scratch. A good first step before tackling convolutional layers specifically.

- [CNN from Scratch — LatinX in AI (Medium)](https://medium.com/latinxinai/convolutional-neural-network-from-scratch-6b1c856e1c07):
  Steps through implementing the convolutional layer including forward and backward passes. Closest to what this project requires.

- [Understanding Neural Networks by Building One — Medium](https://medium.com/@pankajgoyal4152/understanding-neural-networks-by-building-one-from-scratch-a-beginners-journey-3a11617313a4):
  Written from a beginner's perspective with a similar motivation to this project. Useful for understanding the mindset shift required.

- [CNN from Scratch — GitHub (vzhou842)](https://github.com/vzhou842/cnn-from-scratch):
  A compact, well-commented reference implementation. Useful to compare against your own code when debugging.

- [Neural Networks from Scratch — IBM Developer](https://developer.ibm.com/articles/neural-networks-from-scratch/):
  Covers theoretical foundations clearly with code examples. Helpful when you encounter unfamiliar formulas during implementation.

- [ADAM Optimisation — Dive into Deep Learning](https://d2l.ai/chapter_optimization/adam.html):
  Clear mathematical explanation of the ADAM optimiser. Essential reading before implementing your own.

### A note on using AI assistance

For complex mathematical formulas — backpropagation gradients, vectorised convolution operations — using an AI tool to help translate equations into NumPy code is a reasonable and practical approach, as long as you understand what the code is doing before using it. Blindly copying formula implementations will make debugging nearly impossible.

---

## Evaluation

### How useful is it compared to the effort?

Building a CNN from scratch is genuinely difficult and time-consuming. Expect to spend significantly more time debugging numerical issues (vanishing gradients, NaN losses, shape mismatches) than you would using PyTorch. However, the depth of understanding gained is difficult to replicate any other way.

The model built for this project achieved **69.5% accuracy** and a **weighted F1 score of 0.72** on a 7-class skin lesion classification task — demonstrating that a well-implemented scratch model is genuinely functional.

![Learning Curves|385](Confusion%20Matrix.png)
*Confusion matrix across 7 classes. The model performs strongest on `nv` due to it being the most represented class, highlighting how dataset imbalance directly affects per-class performance.*

![Confusion Matrix|430](Learning%20Curve.png)
*Loss and accuracy curves showing training converged at epoch 106. The gap between training and validation loss indicates mild overfitting despite regularisation.*

![GradCAM Classification|568](GradCAM_right.png)
*GradCAM visualisation of a correct classification (true: vasc, predicted: vasc). The
model focuses on a central region where the critical feature is.

![GradCAM Mislassification|568](GradCAM_wrong.png)
*GradCAM visualisation of a misclassification (true: bkl, predicted: akiec). The model focuses on a central region rather than diagnostically relevant features, suggesting image quality and preprocessing remain important factors.*

### Alternatives

If your goal is a production-ready classifier, PyTorch or Keras will get you there far faster. pretrained models via transfer learning will also outperform a from-scratch CNN on most real datasets. Building from scratch is best justified as a **learning exercise** rather than a practical engineering choice.

| Approach | Learning Value | Development Speed | Performance Ceiling |
|---|---|---|---|
| From scratch (NumPy) | ⭐⭐⭐⭐⭐ | Slow | Moderate |
| PyTorch / Keras | ⭐⭐⭐ | Fast | High |
| Transfer Learning | ⭐⭐ | Very Fast | Very High |