import av
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_theme()
import tensorflow as tf
import time
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Start time
t_start = time.time()

A_dataset = av.A_dataset

# --- Begin Autoencoder Definition ---

input_dim = A_dataset.shape[1]  # Aantal features in uw dataset
encoding_dim = 2  # Aantal neuronen in de bottleneck layer (voor 2D-projectie)

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)

# Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder = Encoder + Decoder
autoencoder = Model(input_layer, decoded)

# Aparte encoder (om later de gereduceerde dimensies te kunnen ophalen)
encoder = Model(input_layer, encoded)

# Compileer en train de autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
history = autoencoder.fit(A_dataset, A_dataset, epochs=100, batch_size=256, shuffle=True)

# Gebruik de encoder om de gereduceerde dimensies te verkrijgen
Df_embedding = encoder.predict(A_dataset)
Df_embedding = pd.DataFrame(Df_embedding, columns=['Dimension 1', 'Dimension 2'])

# Bereken de "stressfactor" voor de autoencoder (MSE)
final_mse = history.history['loss'][-1]

# --- Einde Autoencoder Definition ---

# Visualisatie
sns.set_theme('notebook')
sns.set_style('darkgrid')
plt.figure(figsize=(8,8), dpi=100.0)
plot = sns.scatterplot(data=Df_embedding, x='Dimension 1', y='Dimension 2', markers=True, legend="brief", s=50, color='black')
plot.set(xlabel=None)
plot.set(ylabel=None)

ax = plt.gca()  # Get the current axes
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.gca().set_facecolor('white') 
x_ticks = np.arange(-30, 30, 5)
plt.yticks(x_ticks, color='black', fontsize=8)
ax.xaxis.grid(True, linestyle='dotted', linewidth=0.5, color='black', alpha=0.5)

y_ticks = np.arange(-30, 30, 5)
plt.yticks(y_ticks, color='black', fontsize=8)
ax.yaxis.grid(True, linestyle='dotted', linewidth=0.5, color='black', alpha=0.5)

# Loop for annotation of all points
for i in range(len(A_dataset)):
    plt.annotate(i, xy=(Df_embedding.iloc[i, 0], Df_embedding.iloc[i, 1]), xytext=(25, 25), textcoords="offset pixels")

# Opslaan van de visualisatie
filename = 'Autoencoder_Reduction.png'  # Wijzig de bestandsnaam indien nodig
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.clf()

# Print de "stressfactor" (MSE)
print(f'Final MSE (stress factor) for the autoencoder: {final_mse:.5f}')

# End and print time
print('Time elapsed for running module with Autoencoder: {:.3f} sec.'.format(time.time() - t_start))