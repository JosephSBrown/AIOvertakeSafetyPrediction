import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

df = pd.read_json('trainingData.json')

X = df[['speed_vehicle', 'speed_oncoming_vehicle', 'distance_oncoming_vehicle']].values.astype('float32')
y = df['safety']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(64, activation  ='relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
  ]
)

initial_learning_rate = 0.06
decay_steps = 1000
decay_rate = 0.9

learning_rate_schedule = tf.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps, decay_rate)

optimiser = tf.keras.optimizers.Adam(learning_rate = learning_rate_schedule)

model.compile(optimizer = optimiser, loss = 'binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"test accuracy: {test_accuracy}")

speed_vehicle_pred = input("What is Your Vehicle Speed? ")
oncoming_vehicle_pred = input("What is the Oncoming Vehicle's Speed? ")
oncoming_distance_pred = input("What is the Oncoming Vehicle's Distance? ")

new_situation = [[float(speed_vehicle_pred), float(oncoming_vehicle_pred), float(oncoming_distance_pred)]]
situation_tensor = tf.convert_to_tensor(new_situation, dtype=tf.float32)
prediction = model.predict(situation_tensor)
if prediction > 0.5:
    print("It's safe to overtake the vehicle.")
else:
    print("It's not safe to overtake the vehicle.")