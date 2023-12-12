import tensorflow as tf
import pandas as pd

df = pd.read_csv('dataset.csv')

df['Outcome'] = df['Outcome'].map({'Success': 1, 'Failure': 0, 'Close Call': 0, 'Risk': 0}).fillna(0)

features = df[['Speed_of_Vehicle', 'Speed_of_Oncoming_Vehicle', 'Distance_to_Oncoming_Vehicle']]
labels = df['Outcome']

min_values = features.min()
max_values = features.max()
features = (features - min_values) / (max_values - min_values)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.Dense(16, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

initial_learning_rate = 0.004
decay_steps = 8000
decay_rate = 0.9

learning_rate_schedule = tf.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps, decay_rate)

optimiser = tf.keras.optimizers.Adam(learning_rate = learning_rate_schedule)

model.compile(optimizer=optimiser, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(features, labels, epochs=1000, verbose=1)

test_loss, test_accuracy = model.evaluate(features, labels)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

speed_vehicle_pred = float(input("Enter Your Vehicle's Current Speed: "))
oncoming_vehicle_pred = float(input("Enter the Oncoming Vehicle's Current Speed: "))
oncoming_distance_pred = float(input("Enter the Distance to the Oncoming Vehicle: "))

normalized_input = [
    (speed_vehicle_pred - min_values['Speed_of_Vehicle']) / (max_values['Speed_of_Vehicle'] - min_values['Speed_of_Vehicle']),
    (oncoming_vehicle_pred - min_values['Speed_of_Oncoming_Vehicle']) / (max_values['Speed_of_Oncoming_Vehicle'] - min_values['Speed_of_Oncoming_Vehicle']),
    (oncoming_distance_pred - min_values['Distance_to_Oncoming_Vehicle']) / (max_values['Distance_to_Oncoming_Vehicle'] - min_values['Distance_to_Oncoming_Vehicle'])
]

prediction = model.predict([normalized_input])

if prediction[0][0] > 0.5:
    print("It is safe to overtake.")
else:
    print("It is not safe to overtake.")