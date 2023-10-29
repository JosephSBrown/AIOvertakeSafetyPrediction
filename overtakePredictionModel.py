import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

data = [
  {
      "oncomingspeed_mph": 70,
      "oncomingdistance_meters": 50,
      "currentspeed_mph": 60,
      "SpeedOfCarBeingOvertaken_mph": 50,
      "visibility_percentage": 75,
      "rating": 1
  },
  {
      "oncomingspeed_mph": 75,
      "oncomingdistance_meters": 40,
      "currentspeed_mph": 65,
      "SpeedOfCarBeingOvertaken_mph": 45,
      "visibility_percentage": 65,
      "rating": 2
  },
  {
      "oncomingspeed_mph": 60,
      "oncomingdistance_meters": 60,
      "currentspeed_mph": 70,
      "SpeedOfCarBeingOvertaken_mph": 55,
      "visibility_percentage": 85,
      "rating": 0
  },
  {
      "oncomingspeed_mph": 55,
      "oncomingdistance_meters": 70,
      "currentspeed_mph": 75,
      "SpeedOfCarBeingOvertaken_mph": 60,
      "visibility_percentage": 55,
      "rating": 1
  },
  {
      "oncomingspeed_mph": 80,
      "oncomingdistance_meters": 30,
      "currentspeed_mph": 65,
      "SpeedOfCarBeingOvertaken_mph": 70,
      "visibility_percentage": 70,
      "rating": 2
  },
  {
      "oncomingspeed_mph": 65,
      "oncomingdistance_meters": 55,
      "currentspeed_mph": 70,
      "SpeedOfCarBeingOvertaken_mph": 65,
      "visibility_percentage": 75,
      "rating": 0
  },
  {
      "oncomingspeed_mph": 75,
      "oncomingdistance_meters": 45,
      "currentspeed_mph": 65,
      "SpeedOfCarBeingOvertaken_mph": 60,
      "visibility_percentage": 60,
      "rating": 1
  },
  {
      "oncomingspeed_mph": 70,
      "oncomingdistance_meters": 50,
      "currentspeed_mph": 80,
      "SpeedOfCarBeingOvertaken_mph": 70,
      "visibility_percentage": 80,
      "rating": 1
  },
  {
      "oncomingspeed_mph": 85,
      "oncomingdistance_meters": 25,
      "currentspeed_mph": 70,
      "SpeedOfCarBeingOvertaken_mph": 75,
      "visibility_percentage": 70,
      "rating": 2
  },
  {
      "oncomingspeed_mph": 65,
      "oncomingdistance_meters": 55,
      "currentspeed_mph": 65,
      "SpeedOfCarBeingOvertaken_mph": 60,
      "visibility_percentage": 75,
      "rating": 0
  },
  {
    "oncomingspeed_mph": 80,
    "oncomingdistance_meters": 30,
    "currentspeed_mph": 65,
    "SpeedOfCarBeingOvertaken_mph": 70,
    "visibility_percentage": 70,
    "rating": 2
},
{
    "oncomingspeed_mph": 65,
    "oncomingdistance_meters": 55,
    "currentspeed_mph": 70,
    "SpeedOfCarBeingOvertaken_mph": 65,
    "visibility_percentage": 75,
    "rating": 0
},
{
    "oncomingspeed_mph": 75,
    "oncomingdistance_meters": 45,
    "currentspeed_mph": 65,
    "SpeedOfCarBeingOvertaken_mph": 60,
    "visibility_percentage": 60,
    "rating": 2
},
{
    "oncomingspeed_mph": 70,
    "oncomingdistance_meters": 50,
    "currentspeed_mph": 80,
    "SpeedOfCarBeingOvertaken_mph": 70,
    "visibility_percentage": 80,
    "rating": 1
},
{
    "oncomingspeed_mph": 85,
    "oncomingdistance_meters": 25,
    "currentspeed_mph": 70,
    "SpeedOfCarBeingOvertaken_mph": 75,
    "visibility_percentage": 70,
    "rating": 0
},
{
    "oncomingspeed_mph": 65,
    "oncomingdistance_meters": 55,
    "currentspeed_mph": 65,
    "SpeedOfCarBeingOvertaken_mph": 60,
    "visibility_percentage": 75,
    "rating": 1
},
{
    "oncomingspeed_mph": 70,
    "oncomingdistance_meters": 50,
    "currentspeed_mph": 60,
    "SpeedOfCarBeingOvertaken_mph": 50,
    "visibility_percentage": 75,
    "rating": 2
},
{
    "oncomingspeed_mph": 75,
    "oncomingdistance_meters": 40,
    "currentspeed_mph": 65,
    "SpeedOfCarBeingOvertaken_mph": 45,
    "visibility_percentage": 65,
    "rating": 0
},
{
    "oncomingspeed_mph": 60,
    "oncomingdistance_meters": 60,
    "currentspeed_mph": 70,
    "SpeedOfCarBeingOvertaken_mph": 55,
    "visibility_percentage": 85,
    "rating": 1
},
{
    "oncomingspeed_mph": 55,
    "oncomingdistance_meters": 70,
    "currentspeed_mph": 75,
    "SpeedOfCarBeingOvertaken_mph": 60,
    "visibility_percentage": 55,
    "rating": 2
},
{
    "oncomingspeed_mph": 80,
    "oncomingdistance_meters": 30,
    "currentspeed_mph": 65,
    "SpeedOfCarBeingOvertaken_mph": 70,
    "visibility_percentage": 70,
    "rating": 0
},
{
    "oncomingspeed_mph": 65,
    "oncomingdistance_meters": 55,
    "currentspeed_mph": 70,
    "SpeedOfCarBeingOvertaken_mph": 65,
    "visibility_percentage": 75,
    "rating": 1
},
{
    "oncomingspeed_mph": 75,
    "oncomingdistance_meters": 45,
    "currentspeed_mph": 65,
    "SpeedOfCarBeingOvertaken_mph": 60,
    "visibility_percentage": 60,
    "rating": 2
},
{
    "oncomingspeed_mph": 70,
    "oncomingdistance_meters": 50,
    "currentspeed_mph": 80,
    "SpeedOfCarBeingOvertaken_mph": 70,
    "visibility_percentage": 80,
    "rating": 0
},
{
    "oncomingspeed_mph": 85,
    "oncomingdistance_meters": 25,
    "currentspeed_mph": 70,
    "SpeedOfCarBeingOvertaken_mph": 75,
    "visibility_percentage": 70,
    "rating": 1
},
{
    "oncomingspeed_mph": 65,
    "oncomingdistance_meters": 55,
    "currentspeed_mph": 65,
    "SpeedOfCarBeingOvertaken_mph": 60,
    "visibility_percentage": 75,
    "rating": 2
},
{
    "oncomingspeed_mph": 70,
    "oncomingdistance_meters": 50,
    "currentspeed_mph": 60,
    "SpeedOfCarBeingOvertaken_mph": 50,
    "visibility_percentage": 75,
    "rating": 0
},
{
    "oncomingspeed_mph": 75,
    "oncomingdistance_meters": 40,
    "currentspeed_mph": 65,
    "SpeedOfCarBeingOvertaken_mph": 45,
    "visibility_percentage": 65,
    "rating": 1
},
{
    "oncomingspeed_mph": 60,
    "oncomingdistance_meters": 60,
    "currentspeed_mph": 70,
    "SpeedOfCarBeingOvertaken_mph": 55,
    "visibility_percentage": 85,
    "rating": 2
},
{
    "oncomingspeed_mph": 55,
    "oncomingdistance_meters": 70,
    "currentspeed_mph": 75,
    "SpeedOfCarBeingOvertaken_mph": 60,
    "visibility_percentage": 55,
    "rating": 0
},
{
    "oncomingspeed_mph": 80,
    "oncomingdistance_meters": 30,
    "currentspeed_mph": 65,
    "SpeedOfCarBeingOvertaken_mph": 70,
    "visibility_percentage": 70,
    "rating": 1
},
{
    "oncomingspeed_mph": 65,
    "oncomingdistance_meters": 55,
    "currentspeed_mph": 70,
    "SpeedOfCarBeingOvertaken_mph": 65,
    "visibility_percentage": 75,
    "rating": 2
},
{
    "oncomingspeed_mph": 75,
    "oncomingdistance_meters": 45,
    "currentspeed_mph": 65,
    "SpeedOfCarBeingOvertaken_mph": 60,
    "visibility_percentage": 60,
    "rating": 0
},
{
    "oncomingspeed_mph": 70,
    "oncomingdistance_meters": 50,
    "currentspeed_mph": 80,
    "SpeedOfCarBeingOvertaken_mph": 70,
    "visibility_percentage": 80,
    "rating": 1
},
{
    "oncomingspeed_mph": 85,
    "oncomingdistance_meters": 25,
    "currentspeed_mph": 70,
    "SpeedOfCarBeingOvertaken_mph": 75,
    "visibility_percentage": 70,
    "rating": 2
},
{
    "oncomingspeed_mph": 65,
    "oncomingdistance_meters": 55,
    "currentspeed_mph": 65,
    "SpeedOfCarBeingOvertaken_mph": 60,
    "visibility_percentage": 75,
    "rating": 0
},
{
    "oncomingspeed_mph": 75,
    "oncomingdistance_meters": 45,
    "currentspeed_mph": 65,
    "SpeedOfCarBeingOvertaken_mph": 60,
    "visibility_percentage": 60,
    "rating": 1
},
{
    "oncomingspeed_mph": 70,
    "oncomingdistance_meters": 50,
    "currentspeed_mph": 60,
    "SpeedOfCarBeingOvertaken_mph": 50,
    "visibility_percentage": 75,
    "rating": 2
},
{
    "oncomingspeed_mph": 75,
    "oncomingdistance_meters": 40,
    "currentspeed_mph": 65,
    "SpeedOfCarBeingOvertaken_mph": 45,
    "visibility_percentage": 65,
    "rating": 0
},
{
    "oncomingspeed_mph": 60,
    "oncomingdistance_meters": 60,
    "currentspeed_mph": 70,
    "SpeedOfCarBeingOvertaken_mph": 55,
    "visibility_percentage": 85,
    "rating": 1
}
]

df = pd.DataFrame(data)

X = df[['oncomingspeed_mph', 'oncomingdistance_meters', 'currentspeed_mph', 'SpeedOfCarBeingOvertaken_mph', 'visibility_percentage']].values.astype('float32')
y = df['rating']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5,)),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(64, activation  ='relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
  ]
)

initial_learning_rate = 0.01
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
overtaken_vehicle_pred = input("What is the Vehicle Speed of the Vehicle Being Overtaken? ")
visibility_pred = input("What is the Current Visibility Percentage? ")

user_input = {
    "oncoming_speed_mph": oncoming_vehicle_pred,
    "distance_to_oncoming_car_meters": oncoming_distance_pred,
    "current_speed_mph": speed_vehicle_pred,
    "speed_of_car_being_overtaken_mph": overtaken_vehicle_pred,
    "visibility_percentage": visibility_pred
}

user_input_array = np.array([
    user_input["oncoming_speed_mph"],
    user_input["distance_to_oncoming_car_meters"],
    user_input["current_speed_mph"],
    user_input["speed_of_car_being_overtaken_mph"],
    user_input["visibility_percentage"]
], dtype=np.float32)

prediction = model.predict(user_input_array.reshape(1, -1))

# The prediction directly represents the numeric rating
predicted_rating = int(prediction[0][0])  # Assuming a single output neuron for the rating
print(predicted_rating)

if predicted_rating == 1:
    print("Successful")
elif predicted_rating == 0:
    print("Failure")
else:
    print("Risk")