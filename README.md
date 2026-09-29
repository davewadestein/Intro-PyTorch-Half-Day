# Intro to PyTorch Through an MNIST Variational Autoencoder

**Half-day course outline | 4 hours, including a 15-minute break**

## Course overview

Students learn the core PyTorch workflow by building toward one practical outcome: a variational autoencoder (VAE) that generates new handwritten digits from MNIST. Short exercises prepare image tensors, build an encoder and decoder, examine gradients, and assemble a training loop. In the capstone lab, students train the VAE and generate new digits.

The class emphasizes a working mental model and code students can edit. A supplied notebook keeps the project achievable in a half day without requiring a full treatment of the underlying mathematics.

## Audience and preparation

Students should be comfortable reading Python functions and classes, running notebooks, and working with NumPy-style arrays. Prior deep learning experience is not required.

**Instructor preparation:** Provide a tested notebook with PyTorch, torchvision, and Matplotlib available. Preload or cache MNIST. Use a small model, a subset of training images, and a modest number of epochs so the exercises can run on a CPU. GPU or Apple MPS support is optional.

## Learning objectives

By the end of the class, students will be able to:

* Create tensors from MNIST images and explain image and batch shapes
* Use a dataset and data loader to create shuffled batches
* Define a model with an encoder and decoder using `nn.Module`
* Explain the purpose of sampling in a VAE
* Describe reconstruction loss and KL divergence as two parts of the loss
* Use gradients and an optimizer to train a model
* Generate and display new digits by sampling the latent space
* Distinguish a reconstruction from a newly generated image

## Course sequence

### 1. See the destination | 15 minutes

Show an MNIST image, its reconstruction, and a newly generated digit. Trace the path from image to encoder, latent description, decoder, and output pixels. Introduce the project students will complete.

**Practice:** Identify which output is a reconstruction and which is a new sample. Sketch the flow through the VAE.

### 2. Images as tensors | 25 minutes

Load MNIST with torchvision. Inspect the shape, data type, and pixel values of one image and a batch. Flatten a 28 × 28 image into 784 values, then restore its shape. Briefly compare PyTorch tensors with NumPy arrays and introduce device placement.

**Practice:** Inspect and plot images, flatten a batch, restore an image, and check that pixel values fall between 0 and 1.

### 3. Build the data pipeline | 30 minutes

Use `Dataset` and `DataLoader` to form batches and shuffle training images. Explain what an epoch is and why training works through batches.

**Practice:** Complete a data loader exercise and confirm the shape of a batch before passing it to the model.

### 4. Build an encoder and decoder | 35 minutes

Introduce `nn.Module`, `forward()`, `nn.Linear`, and learnable parameters. The encoder produces a center and spread for each image’s two-number latent description. The decoder turns a latent point into 784 pixel outputs. Explain sampling as adding appropriately scaled random noise.

**Practice:** Complete the sampling method, run a forward pass, check output shapes, and view an untrained reconstruction.

### 5. Loss, gradients, and an update | 40 minutes

Compare the original pixels with the model’s output using reconstruction loss. Add KL divergence to encourage an organized latent space. Explain how `backward()` computes gradients and how the optimizer updates model parameters.

**Practice:** Carry out one optimizer step and inspect total loss, reconstruction loss, KL divergence, and a parameter gradient.

### Break | 10 minutes

### 6. Assemble the training loop | 25 minutes

Put the steps in order: get a batch, run the model, calculate loss, clear old gradients, calculate new gradients, and update parameters. Track losses across epochs. Introduce `train()`, `eval()`, and `torch.no_grad()`.

**Practice:** Complete the missing lines of the training loop and test it on one batch.

### Capstone lab: Train and generate digits | 50 minutes

Train the VAE on a subset of MNIST. Compare original images with reconstructions, then choose new points in the latent space and decode them into digits. Plot losses by epoch and discuss the results.

**Practice:** Finish the notebook, generate a grid of new digits, and try one change, such as training for more epochs or changing the latent dimension.

### Wrap-up | 10 minutes

Recap the **tensor → model → loss → gradient → optimizer** workflow. Ask students where the decoder’s input comes from during reconstruction versus generation, and what experiment they would try next.

## Scope notes

This is a guided introduction to a VAE. The notebook supplies the KL divergence formula and focuses on what each part does. Convolutional layers, mathematical derivations, deployment, and extensive parameter tuning are outside the scope of the class. For slower machines, use a short training run in class and provide a saved model for the generation exercises.
