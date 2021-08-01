import keras
import numpy as np
from matplotlib import pyplot as plt


def generater_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


x_gan = generater_latent_points(100, 25)

for j in range(10):
    model = keras.models.load_model('generator_model_0'+str(j)+'1.h5')

    out = model.predict_on_batch(x_gan)

    for i in range(out.shape[0]):
        plt.subplot(5,5,i+1)
        plt.axis('off')
        plt.imshow(out[i,:,:,0],cmap='gray_r')
    plt.show()