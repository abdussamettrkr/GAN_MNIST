import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense,Reshape,Conv2DTranspose
from keras.optimizers import Adam


def loadTrainData():
    (train_X, _), (_, _) = mnist.load_data()
    # expand data to 3d
    train_X = np.expand_dims(train_X, axis=-1)
    ##normilize data
    train_X = train_X.astype(dtype='float32')
    train_X = train_X / 255.0
    return train_X


def create_Discrimator(input_shape=(28, 28, 1)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), 2, padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), strides=2, padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    return model

def create_Generator(latent_dim):
    model = Sequential()
    ##
    n_nodes = 128*7*7
    model.add(Dense(n_nodes,input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7,7,128)))
    model.add(Conv2DTranspose(128,(4,4),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128,(4,4),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1,(7,7),activation='sigmoid',padding='same'))
    return model


def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, 1))
    return X, y

def generate_fake_samples(g_model,latent_dim,n_samples):
    x_input= generater_latent_points(latent_dim,n_samples)
    X = g_model.predict(x_input)
    y = np.zeros((n_samples,1))
    return X,y

def generater_latent_points(latent_dim,n_samples):
    x_input = np.random.randn(latent_dim*n_samples)
    x_input = x_input.reshape(n_samples,latent_dim)
    return x_input


def train_discriminator(model,dataset,n_iter=100,n_batch=256):
    half_batch = n_batch//2

    for i in range(n_iter):
        x_real,y_real = generate_real_samples(dataset,half_batch)
        x_fake,y_fake = generate_fake_samples(half_batch)

        _,real_acc = model.train_on_batch(x_real,y_real)
        _, fake_acc = model.train_on_batch(x_fake, y_fake)

        print('%d real=%.2f fake=%.2f' % (i+1,real_acc,fake_acc))

def create_gan(d_model,g_model):
    model = Sequential()
    d_model.trainable = False

    model.add(g_model)
    model.add(d_model)
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0002,beta_1 = 0.5))

    return model

def train(g_model,d_model,gan_model,dataset,latent_dim,n_epochs=91,n_batch=256):
    batch_per_epoch = dataset.shape[0]//n_batch
    half_batch= n_batch//2

    for i in range(n_epochs):
        for j in range(batch_per_epoch):
            x_real,y_real = generate_real_samples(dataset,half_batch)
            x_fake,y_fake = generate_fake_samples(g_model,latent_dim,half_batch)

            X,y = np.vstack((x_real,x_fake)),np.vstack((y_real,y_fake))

            d_loss,_ =   d_model.train_on_batch(X,y)

            X_gan = generater_latent_points(latent_dim,n_batch)

            y_gan = np.ones((n_batch,1))
            g_loss = gan_model.train_on_batch(X_gan,y_gan)

            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i + 1, j + 1, batch_per_epoch, d_loss, g_loss))

        if i %10 == 0:
            filename = 'generator_model_%03d.h5' % (i + 1)
            g_model.save(filename)

generator = create_Generator(100)
discriminator = create_Discrimator()
model = create_gan(discriminator,generator)

dataset = loadTrainData()
latent_dim = 100
train(generator,discriminator,model,dataset,latent_dim)
