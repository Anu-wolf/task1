import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


path = 'student-performance-factors.csv'
student_performance = pd.read_csv(path)
print(student_performance.head())

# Define features and target
y = student_performance.Exam_Score
features = [
    'Hours_Studied', 'Previous_Scores', 'Attendance', 'Sleep_Hours', 'Tutoring_Sessions',
    'Physical_Activity', 'Parental_Involvement', 'Gender', 'Access_to_Resources', 'Extracurricular_Activities',
    'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 'School_Type',
    'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level', 'Distance_from_Home'
]
X = student_performance[features]
X = pd.get_dummies(X)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Normalize data
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
val_X = scaler.transform(val_X)

# Reshaping
train_X = train_X.reshape(train_X.shape[0], 40, 1, 1)
val_X = val_X.reshape(val_X.shape[0], 40, 1, 1)

# target to categorical for multiclass classification
num_classes = train_y.max() + 1  # Determine the number of unique classes
train_y_cat = tf.keras.utils.to_categorical(train_y, num_classes=num_classes)  
val_y_cat = tf.keras.utils.to_categorical(val_y, num_classes=num_classes)  

# MODEL ONE
model = models.Sequential([
    Input(shape=(40, 1, 1)),
    layers.Conv2D(32, (2, 1), activation='relu'),
    layers.MaxPooling2D((2, 1)),
    layers.Conv2D(64, (2, 1), activation='relu'),
    layers.MaxPooling2D((2, 1)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')   # Softmax for multiclass 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_X, 
    train_y_cat,  #  one-hot encoded labels 
    epochs=90, 
    batch_size=32,
    validation_data=(val_X, val_y_cat)  # One-hot encoded validation labels
)

# MODEL TWO
model2 = models.Sequential([
    Input(shape=(40, 1, 1)),
    layers.Conv2D(32, (2, 1), activation='relu'),
    layers.MaxPooling2D((2, 1)),
    layers.Conv2D(64, (2, 1), activation='relu'),
    layers.MaxPooling2D((2, 1)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')   # Softmax for multiclass 
])

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history2 = model2.fit(
    train_X, 
    train_y_cat,  # one-hot encoded labels 
    epochs=90, 
    batch_size=32,
    validation_data=(val_X, val_y_cat)  # One-hot encoded validation labels
)

# Evaluation
y_pred_1 = model.predict(val_X)
y_pred_1 = np.argmax(y_pred_1, axis=1)  # Conversion one-hot encoded predictions to class labels

y_pred_2 = model2.predict(val_X)
y_pred_2 = np.argmax(y_pred_2, axis=1)  

# metrics 
accuracy_1 = accuracy_score(val_y, y_pred_1)
precision_1 = precision_score(val_y, y_pred_1, average='macro')  # macro for multiclass
f1_1 = f1_score(val_y, y_pred_1, average='macro')

accuracy_2 = accuracy_score(val_y, y_pred_2)
precision_2 = precision_score(val_y, y_pred_2, average='macro')
f1_2 = f1_score(val_y, y_pred_2, average='macro')

# Print results
print("Model 1 Performance:")
print(f"Accuracy: {accuracy_1:.4f}, Precision: {precision_1:.4f}, F1 Score: {f1_1:.4f}")

print("\nModel 2 Performance:")
print(f"Accuracy: {accuracy_2:.4f}, Precision: {precision_2:.4f}, F1 Score: {f1_2:.4f}")
