import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Charlie Misbach
# Machine Learning - CSCI3470
# Assignment #2
# Due: 3/10/25

# Question #1: Download MNIST and create subplots of 
# normalized grayscaled images

# Convert the image to a tensor of floats in [0, 1] (pixel values)
transform = transforms.Compose([
    transforms.ToTensor()  # converts [0,255] -> [0,1] (pixel values)
])

# Download AND load the MNIST training dataset with defined transform
train_dataset = torchvision.datasets.MNIST(
    root='./data',    # folder to download/extract
    train=True,
    transform=transform,
    download=True
)

# Download AND load MNIST test dataset
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

# Create a dataloader for the training dataset
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

# Create 4x4 grid of subplots to display 16 images
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(6,6))
axes = axes.flatten()

# loop through all 16 samples of the training set
for i in range(16):
    img, label = train_dataset[i]  # each item is (image_tensor, label)
    
    # img is a tensor shape [1,28,28] (grayscale). Convert to numpy for plotting:
    img_np = img.squeeze().numpy()  
    
    # Display image
    axes[i].imshow(img_np, cmap='gray')
    axes[i].set_title(f"Label: {label}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Question #2: Implement LeNet-5 with Keras functional API
import tensorflow as tf
from tensorflow import keras
from keras import layers

# LeNet-5 (classic CNN arch) consists of 
# 1. Convolutional Layers (feature extraction)
# 2. Subsampling (Pooling) layers (Reduces dimensions)
# 3. Fully connected layers (classification)

# Define the input shape for MNIST images (28x28 grayscale)
inputs = keras.Input(shape=(28, 28, 1))

# First Convolutional Layer (6 filters, 5x5 kernel, ReLU activation)
x = layers.Conv2D(filters=6, kernel_size=(5,5), activation="relu", padding="same")(inputs)
x = layers.AveragePooling2D(pool_size=(2,2))(x)  # Subsampling (Average Pooling)

# Second Convolutional Layer: 16 filters, 5x5 kernel, ReLU activation
x = layers.Conv2D(filters=16, kernel_size=(5,5), activation="relu")(x)
x = layers.AveragePooling2D(pool_size=(2,2))(x)  # Subsampling layer

# Flatten the extracted features before passing them into Dense layers
x = layers.Flatten()(x)

# Fully Connected Layer: 120 neurons
x = layers.Dense(units=120, activation="relu")(x)

# Fully Connected Layer: 84 neurons
x = layers.Dense(units=84, activation="relu")(x)

# Output Layer: 10 neurons for digit classification (Softmax activation)
outputs = layers.Dense(units=10, activation="softmax")(x)

# Create the Model
lenet_model = keras.Model(inputs=inputs, outputs=outputs, name="LeNet-5")

# Print the model structure
lenet_model.summary()

# Compile the model
lenet_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"]
)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize images to [0,1] range
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape images to add the channel dimension (grayscale → (28,28,1))
x_train = x_train[..., None]
x_test = x_test[..., None]

# Process images in batches of 64
# Train for 10 epochs (10 passes through entire dataset)
# Use 20% of the train data for validation data (throughout the training)
training = lenet_model.fit(
    x_train, y_train,
    batch_size=64,    # Process images in batches of 64
    epochs=10,        # Train for 10 epochs
    validation_split=0.2  # Use 20% of training data for validation
)

test_scores = lenet_model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

# Question #3: Train the implemented LeNet5 model and plotting the
# training results (for test and validation)

# Extract loss values from training history
train_loss = training.history['loss']  # Training loss
val_loss = training.history['val_loss']  # Validation loss
epochs = range(1, len(train_loss) + 1)  # Epoch numbers

# Plot Training vs Validation Loss
plt.figure(figsize=(8,6))
plt.plot(epochs, train_loss, 'b-', label='Training Loss')   # Blue Line - Training Loss
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')   # Red Line - Validation Loss
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss')
plt.legend()
plt.grid()
plt.show()