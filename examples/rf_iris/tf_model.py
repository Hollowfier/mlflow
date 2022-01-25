
import tensorflow as tf
 
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [5, 8, 11, 14, 17, 20, 23, 26, 29, 32]
 
# Define layer
layer0 = tf.keras.layers.Dense(units=1, input_shape=[1])
 
model = tf.keras.Sequential([layer0])
 
# Compile model
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(1))
 
# Train the model
history = model.fit(x, y, epochs=100, verbose=False)
 
# Prediction
print('Prediction: {}'.format(model.predict([10])))
 
# Get weight and bias
weights = layer0.get_weights()
print('weight: {} bias: {}'.format(weights[0], weights[1]))

model.save('E:/eim_ase/mlflow_git/examples/rf_iris/my_tf_model')
print("model saved")
