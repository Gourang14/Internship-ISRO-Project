#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np 
import pandas as pd
import os 
import tensorflow as tf
from tqdm import tqdm


# DEFINING ANGLES AND LABEL FRAMES 

# In[ ]:


# Define the range and step of angles
start_angle = 0.0
end_angle = 180.0
step = 0.5

angles = np.arange(start_angle, end_angle, step)


# In[ ]:


# Paths
source_directory = r"C:\Users\my\Desktop\ml and ai\frames"
destination_directory = r"C:\Users\my\Desktop\ml and ai\output_frames"
ground_truth_csv = r"C:\Users\my\Desktop\ground_truth_velocities.csv"

# Make sure destination directory exists
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Read ground truth CSV
ground_truth = pd.read_csv(ground_truth_csv)

# Iterate over the rows in the CSV
for index, row in ground_truth.iterrows():
    frame_filename = row['frame_filename']
    angle = row['angle']
    
    # Create a directory for each angle
    label_dir = os.path.join(r"C:\Users\my\Desktop\ml and ai\output_frames", f"{angle}_degrees")
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    # Move the frame to the labeled directory
    source_path = os.path.join(source_directory, frame_filename)
    destination_path = os.path.join(label_dir, frame_filename)
    
    if os.path.exists(source_path):
        os.rename(source_path, destination_path)
    else:
        print(f"Frame {frame_filename} not found in source directory.")


# In[ ]:


import shutil

# Paths - update these with the correct paths on your system
source_directory = r"C:\Users\my\Desktop\ml and ai\frames"
destination_directory = r"C:\Users\my\Desktop\ml and ai\output_frames"
ground_truth_velocity_csv = r"C:\Users\my\Desktop\ground_truth_velocities.csv"

# Ensure destination directory exists
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Read ground truth CSV with velocities
ground_truth = pd.read_csv(ground_truth_velocity_csv)

# Iterate over the rows in the CSV
for index, row in ground_truth.iterrows():
    frame_filename = row['frame_filename']
    velocity = row['velocity']
    
    # Create a directory for each velocity
    label_dir = os.path.join(destination_directory, f"{velocity:.2f}_m_s")
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    # Copy the frame to the labeled directory
    source_path = os.path.join(source_directory, frame_filename)
    destination_path = os.path.join(label_dir, frame_filename)
    
    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
        os.remove(source_path)  # Remove the file from the source after copying
    else:
        print(f"Frame {frame_filename} not found in source directory.")


# Conversion of Angles to Velocity

# In[ ]:


# Define your constant k
k = 1.0

# Path to the CSV file with angle labels
ground_truth_csv = 'ground_truth_angles.csv'
# Path to save the new CSV file with velocity labels
ground_truth_velocity_csv = 'ground_truth_velocities.csv'

# Read the CSV file with angles
ground_truth = pd.read_csv(ground_truth_csv)

# Function to convert angle to velocity
def angle_to_velocity(angle, k):
    angle_rad = np.radians(angle)  # Convert angle to radians
    return k * np.tan(angle_rad)

# Apply the conversion to each angle in the DataFrame
ground_truth['velocity'] = ground_truth['angle'].apply(lambda angle: angle_to_velocity(angle, k))

# Save the new DataFrame with velocities
ground_truth.to_csv(ground_truth_velocity_csv, index=False)

print(f"Converted angles to velocities and saved to {ground_truth_velocity_csv}")


# Preprocessing 

# In[ ]:


# Paths
labeled_frames_directory = 'path_to_labeled_frames'
preprocessed_frames_directory = 'path_to_preprocessed_frames'

# Ensure the preprocessed directory exists
if not os.path.exists(preprocessed_frames_directory):
    os.makedirs(preprocessed_frames_directory)

# Iterate over the labeled directories
for label_dir in os.listdir(labeled_frames_directory):
    current_label_dir = os.path.join(labeled_frames_directory, label_dir)
    save_label_dir = os.path.join(preprocessed_frames_directory, label_dir)
    
    if not os.path.exists(save_label_dir):
        os.makedirs(save_label_dir)
    
    for frame_filename in tqdm(os.listdir(current_label_dir)):
        frame_path = os.path.join(current_label_dir, frame_filename)
        
        # Read the image
        image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize to 128x128 pixels
        resized_image = cv2.resize(image, (128, 128))
        
        # Compute the 2D Fourier transform of the image
        f_transform = np.fft.fft2(resized_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift))
        
        # Normalize the magnitude spectrum
        normalized_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        
        # Save the preprocessed image
        save_path = os.path.join(save_label_dir, frame_filename)
        cv2.imwrite(save_path, normalized_spectrum)


# In[ ]:


def preprocess_images(input_folder, output_folder, size=(128, 128)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for angle_dir in os.listdir(input_folder):
        full_angle_dir = os.path.join(input_folder, angle_dir)
        output_angle_dir = os.path.join(output_folder, angle_dir)
        
        if not os.path.exists(output_angle_dir):
            os.makedirs(output_angle_dir)
        
        for filename in os.listdir(full_angle_dir):
            img_path = os.path.join(full_angle_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, size)
            output_path = os.path.join(output_angle_dir, filename)
            cv2.imwrite(output_path, img_resized)

preprocess_images("labeled_frames", "preprocessed_frames")


# In[ ]:


def load_images_and_labels(base_path):
    images = []
    labels = []
    for label in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, label)):
            for img_file in os.listdir(os.path.join(base_path, label)):
                img_path = os.path.join(base_path, label, img_file)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize((128, 128))
                images.append(np.array(img))
                labels.append(float(label))
    return np.array(images), np.array(labels)

# Load dataset
base_path = 'path_to_dataset'  # Replace with the actual path
images, labels = load_images_and_labels(base_path)

# Normalize images
images = images / 255.0
images = images[..., np.newaxis]  # Add channel dimension

# Convert labels to categorical
labels = (labels * 2).astype(int)  # Convert angles to classes
labels = tf.keras.utils.to_categorical(labels, num_classes=360)

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)


# Training

# In[ ]:


def load_dataset(data_directory, img_size=(128, 128), batch_size=32):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    dataset = datagen.flow_from_directory(
        data_directory,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return dataset


train_dataset = load_dataset("preprocessed_frames", img_size=(128, 128), batch_size=32)


# CNN Model

# In[ ]:


from tensorflow.keras import layers, models

def build_model(input_shape=(128, 128, 1), num_classes=360):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (7, 7), activation='leaky_relu', input_shape=input_shape))
    model.add(layers.Conv2D(32, (5, 5), activation='leaky_relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (5, 5), activation='leaky_relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='leaky_relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (5, 5), activation='leaky_relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='leaky_relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(512, (5, 5), activation='leaky_relu'))
    model.add(layers.Conv2D(512, (3, 3), activation='leaky_relu'))
    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model(input_shape=(128, 128, 1), num_classes=360)
model.summary()


# Compiling the Model

# In[ ]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Training the model

# In[ ]:


history = model.fit(X_train, y_train, epochs=6, batch_size=32, validation_data=(X_val, y_val))


# Evaluation

# In[ ]:


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(X_val,  y_val, verbose=2)
print(f"Validation accuracy: {test_acc}")

