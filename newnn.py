import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import plot_model


data = pd.read_csv('gravity_thrust_failure_dataset.csv')

features = ['gravity', 'velocity', 'sensor_noise', 'motor_temp', 'motor_current', 'motor_speed']
X = data[features].values
y_thrust = data['thrust'].values
y_velocity = data['velocity'].values
y_failure = data['failure_flag'].values
y_emergency = data['emergency_thrust'].values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_thrust_train, y_thrust_test = train_test_split(X_scaled, y_thrust, test_size=0.2, random_state=42)
_, _, y_velocity_train, y_velocity_test = train_test_split(X_scaled, y_velocity, test_size=0.2, random_state=42)
_, _, y_failure_train, y_failure_test = train_test_split(X_scaled, y_failure, test_size=0.2, random_state=42)
_, _, y_emergency_train, y_emergency_test = train_test_split(X_scaled, y_emergency, test_size=0.2, random_state=42)


inputs = Input(shape=(X.shape[1],))
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)

thrust_out = Dense(1, name='thrust')(x)
velocity_out = Dense(1, name='velocity')(x)
failure_out = Dense(1, activation='sigmoid', name='failure_flag')(x)
emergency_out = Dense(1, name='emergency_thrust')(x)

model = Model(inputs=inputs, outputs=[thrust_out, velocity_out, failure_out, emergency_out])
model.compile(
    optimizer='adam',
    loss={
        'thrust': 'mse',
        'velocity': 'mse',
        'failure_flag': 'binary_crossentropy',
        'emergency_thrust': 'mse'
    },
    metrics={
        'failure_flag': 'accuracy'
    }
)


history = model.fit(
    X_train,
    [y_thrust_train, y_velocity_train, y_failure_train, y_emergency_train],
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    verbose=1
)


model.evaluate(X_test, [y_thrust_test, y_velocity_test, y_failure_test, y_emergency_test])


predictions = model.predict(X_test)


plt.plot(history.history['loss'], label='Total Loss')
plt.plot(history.history['val_loss'], label='Val Total Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


plt.plot(history.history['failure_flag_accuracy'], label='Failure Detection Accuracy')
plt.title('Failure Prediction Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


plt.scatter(y_thrust_test, predictions[0], label='Thrust')
plt.xlabel('Actual Thrust')
plt.ylabel('Predicted Thrust')
plt.title('Thrust Prediction')
plt.grid(True)
plt.show()


plt.scatter(y_emergency_test, predictions[3], color='orange', label='Emergency Thrust')
plt.xlabel('Actual Emergency Thrust')
plt.ylabel('Predicted Emergency Thrust')
plt.title('Emergency Thrust Prediction')
plt.grid(True)
plt.show()
