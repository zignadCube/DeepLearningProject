# Denoising Images with Auto Encoders
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

clean_train_path = "dataset/hmnist_28_28_RGB_train.csv"
clean_test_path = "dataset/hmnist_28_28_RGB_test.csv"

noisy_train_path = "dataset/hmnist_28_28_RGB_train_0e055.csv"
noisy_test_path = "dataset/hmnist_28_28_RGB_test_0e055.csv"


def model(width, height):
    # define the input layer with the fixed dimension we used for processing images
    input_layer = tf.keras.layers.Input(shape=(height, width, 3))

    # encoding
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # decoding
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)

    # define the output layer
    output_layer = tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same')(x)

    # create the model with the defines input and output layers
    model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

    # compile the model with "adam" optimizer and "mse" loss
    model.compile(optimizer='adam', loss='mse')

    return model


autoencoder = model(28, 28)



# Load Noisy Images
clean_train_df = pd.read_csv(clean_train_path)
clean_test_df = pd.read_csv(clean_test_path)

noisy_train_df = pd.read_csv(noisy_train_path)
noisy_test_df = pd.read_csv(noisy_test_path)

# Split into X and Y
x_train_clean = clean_train_df.drop(columns=['label'])
y_train_clean = clean_train_df['label']

x_test_clean = clean_test_df.drop(columns=['label'])
y_test_clean = clean_test_df['label']

x_train_noisy = noisy_train_df.drop(columns=['label'])
y_train_noisy = noisy_train_df['label']

x_test_noisy = noisy_test_df.drop(columns=['label'])
y_test_noisy = noisy_test_df['label']

# Reshape
x_train_clean = np.array(x_train_clean).reshape(-1,28,28,3)
x_test_clean = np.array(x_test_clean).reshape(-1,28,28,3)

x_train_noisy = np.array(x_train_noisy).reshape(-1,28,28,3)
x_test_noisy = np.array(x_test_noisy).reshape(-1,28,28,3)

# Train Autoencoder
callback = tf.keras.callbacks.ModelCheckpoint(filepath="AE_denosing_2.h5",
                                              monitor='val_loss', 
                                              mode='min',
                                              verbose=1,
                                              save_best_only=True)
autoencoder.fit(x_train_noisy, x_train_clean, epochs=20, batch_size=16, shuffle=True, validation_data=(x_test_noisy, x_test_clean), callbacks=[callback])

# Denoise Images
n = 10
x_test_denoised = autoencoder.predict(x_test_noisy[:n])

plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_clean[i]*0.5/255+0.5)
    plt.title("original")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display noisy
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i]*0.5/255+0.5)
    plt.title("noisy")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display denoised
    ax = plt.subplot(3, n, i + 1 + n + n)
    plt.imshow(x_test_denoised[i]*0.5/255+0.5)
    plt.title("denoised")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

 