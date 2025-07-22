import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Paths
train_dir = 'C:/Users/Harshitha/Desktop/Projects/fruit_quality_detection/mini_fruits_dataset/train'
test_dir = 'C:/Users/Harshitha/Desktop/Projects/fruit_quality_detection/mini_fruits_dataset/test'

# Parameters
img_size = 224 #Image size for MobileNetV2
batch_size = 32 #Batch size of training
num_classes = 6 #Number of classes in the dataset

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel value to [0, 1]

    horizontal_flip=True, # Randomly flip images
    rotation_range=20, # Randomly rotate images
    zoom_range=0.2 #Randomly zoom images
)

test_datagen = ImageDataGenerator(rescale=1./255) 

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'  # <-- for multi-class
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load pretrained MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = True

# Build top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10
)

# Save model
model.save("fruit_quality_multiclass_model.h5")
