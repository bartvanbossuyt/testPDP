"""
FUNCTIONALITY
    Visual Analytics: Creates a dimension reduction using an autoencoder
EXPLANATION
    Uses a neural network autoencoder for dimensionality reduction of the distance matrix
INPUT
    av.A_dataset (distance matrix)
OUTPUT
    Visualization saved as Autoencoder_Reduction.png
"""

import av
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()
import tensorflow as tf
import time
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Start time
t_start = time.time()

A_dataset = av.A_dataset

# --- Begin Autoencoder Definition ---

input_dim = A_dataset.shape[1]  # Number of features in the dataset
encoding_dim = 2  # Number of neurons in the bottleneck layer (for 2D projection)

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)

# Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder = Encoder + Decoder
autoencoder = Model(input_layer, decoded)

# Separate encoder (to obtain the reduced dimensions later)
encoder = Model(input_layer, encoded)

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
history = autoencoder.fit(A_dataset, A_dataset, epochs=100, batch_size=256, shuffle=True)

# Use the encoder to obtain the reduced dimensions
Df_embedding = encoder.predict(A_dataset)
Df_embedding = pd.DataFrame(Df_embedding, columns=['Dimension 1', 'Dimension 2'])

# Calculate the "stress factor" for the autoencoder (MSE)
final_mse = history.history['loss'][-1]

# --- End Autoencoder Definition ---

# Visualization
sns.set_theme('notebook')
sns.set_style('darkgrid')
plt.figure(figsize=(8, 8))
plot = sns.scatterplot(data=Df_embedding, x='Dimension 1', y='Dimension 2', markers=True, legend="brief", s=50, color='black')
plot.set(xlabel=None)
plot.set(ylabel=None)

ax = plt.gca()
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

# Save the visualization
filename = 'Autoencoder_Reduction.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.clf()

# Print the "stress factor" (MSE)
print(f'Final MSE (stress factor) for the autoencoder: {final_mse:.5f}')

# End and print time
print('Time elapsed for running module with Autoencoder: {:.3f} sec.'.format(time.time() - t_start))
