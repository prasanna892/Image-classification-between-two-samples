import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagenerator = ImageDataGenerator(rescale=1./255)
test_datagenerator = ImageDataGenerator(rescale=1./255)

dir_train, dir_test=input("Train folder directory: "), input("Test folder directory: ")

train_datagenerator = train_datagenerator.flow_from_directory(
    dir_train,
    target_size=(128,128),
    batch_size=40,
    class_mode='binary')

test_datagenerator = test_datagenerator.flow_from_directory(
    dir_test,
    target_size=(128,128),
    batch_size=10,
    class_mode='binary')


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3),padding='same', activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D((2,2),2),
    
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2),2),     
     
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2),2),   
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())

model.compile(loss='binary_crossentropy',
             optimizer=tf.keras.optimizers.Adam(0.001),
             metrics=['accuracy'])

DESIRED_ACCURACY = 1

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    pass

callbacks = myCallback()


model.fit(
    train_datagenerator,
    epochs=100,
    validation_data = test_datagenerator,
    callbacks = [callbacks]
    )


model.save('TrainedSet.h5')