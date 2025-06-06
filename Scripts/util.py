import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

def make_gradcam_heatmap(x_one_sample, model, npy_file, png_file):
  # We get the output of the last convolution layer. We then create a model that goes up to only that layer.
  LAST_CONV_LAYER_NAME = "conv2d_3"
  LAYERS_AFTER_LAST_CONV = ["batch_normalization_3", "re_lu_3", 'max_pooling2d', 'flatten', 'dense']

  last_conv_layer = model.get_layer(LAST_CONV_LAYER_NAME)
  last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

  # Then, we create a model which then takes the output of the model above, and uses the remaining layers to get the final predictions.
  classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
  x = classifier_input
  for layer_name in LAYERS_AFTER_LAST_CONV:
    x = model.get_layer(layer_name)(x)
  classifier_model = tf.keras.Model(classifier_input, x)

  # First, we get the output from the model up till the last convolution layer. We ask tf to watch this tensor output, as we want to calculate the gradients of the predictions of our target class wrt to the output of this model (last convolution layer model).
  with tf.GradientTape() as tape:
    # Convert x_one_sample to a tensor
    # inputs = tf.convert_to_tensor(x_one_sample)
    inputs = x_one_sample
    last_conv_layer_output = last_conv_layer_model(inputs)
    tape.watch(last_conv_layer_output)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]

  grads = tape.gradient(top_class_channel, last_conv_layer_output)
  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

  last_conv_layer_output = last_conv_layer_output.numpy()[0]
  pooled_grads = pooled_grads.numpy()
  for i in range(pooled_grads.shape[-1]):
      last_conv_layer_output[:, :, i] *= pooled_grads[i]

  gradcam = np.mean(last_conv_layer_output, axis=-1)

  gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam)
  gradcam = cv2.resize(gradcam, (242, 50))

  
  np.save(npy_file, gradcam)
  plt.imsave(png_file, gradcam, cmap='jet')
  # plt.imshow(inputs[0, :, :, 0], cmap='jet', alpha=0.5)
  # plt.imshow(gradcam, alpha=0.5)
  # return gradcam

  return top_pred_index, top_class_channel


def plot_prediction_confidence_distribution(numbers1, numbers2, lim=3, bw=0.1, title="test", x="Prediction Confidence", y="Density", output=None):
    # Increase fonts
    plt.rcParams.update({'font.size': 14})
    
    sns.kdeplot(numbers1, fill=False, bw_adjust=bw, color='blue')
    sns.kdeplot(numbers2, fill=False, bw_adjust=bw, color="orange")
    plt.hist(numbers1, histtype='stepfilled', alpha=0.2, density=True, color='blue', label="Correct Prediction")
    plt.hist(numbers2, histtype='stepfilled', alpha=0.2, density=True, color="orange", label="Wrong Predictions")
    plt.title(title)
    # set font of xlabel
    plt.xlabel(x, fontsize=14) 
    plt.xticks(fontsize=14)
    plt.xlabel(x)
    plt.ylabel(y)
    #plt.ylim(0,lim)
    plt.legend(loc = 'upper left')
    if output is not None:
      plt.savefig(output, format = 'png')
  
    plt.show()

def batch_make_gradcam_heatmap(X, model):
  LAST_CONV_LAYER_NAME = "conv2d_3"
  LAYERS_AFTER_LAST_CONV = ["batch_normalization_3", "re_lu_3", 'max_pooling2d', 'flatten', 'dense']

  last_conv_layer = model.get_layer(LAST_CONV_LAYER_NAME)
  last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

  # Then, we create a model which then takes the output of the model above, and uses the remaining layers to get the final predictions.
  classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
  x = classifier_input
  for layer_name in LAYERS_AFTER_LAST_CONV:
    x = model.get_layer(layer_name)(x)
  classifier_model = tf.keras.Model(classifier_input, x)

  # First, we get the output from the model up till the last convolution layer. We ask tf to watch this tensor output, as we want to calculate the gradients of the predictions of our target class wrt to the output of this model (last convolution layer model).
  with tf.GradientTape() as tape:
    inputs = X
    last_conv_layer_output = last_conv_layer_model({'input_layer': inputs})
    tape.watch(last_conv_layer_output)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds, axis=1)
    # top_pred_index = tf.argmax(preds[0])
    # top_class_channel = preds[:, top_pred_index]
    top_class_channel = tf.reduce_max(preds, axis=1) 


  # Calcula os gradientes das ativações da camada de interesse
  grads = tape.gradient(top_class_channel, last_conv_layer_output)

  # Média global espacial dos gradientes
  pooled_grads = tf.reduce_mean(grads, axis=(1, 2)) # deixa os batches e os canais do filtro que serã ponderados mais na frente

  # Pondera as ativações
  weighted_outputs = last_conv_layer_output * pooled_grads[:, tf.newaxis, tf.newaxis, :]  # (B, H', W', C)

  # Soma sobre os canais
  heatmaps = tf.reduce_mean(weighted_outputs, axis=-1)  # (B, H', W')

  # ReLU + normalização
  heatmaps = tf.nn.relu(heatmaps)
  heatmaps /= tf.reduce_max(heatmaps, axis=(1, 2), keepdims=True) + 1e-8

  return heatmaps, top_pred_index, top_class_channel




def amostragem_estratificada_indices(labels, n_amostras=20, random_state=42):
    """
    Retorna índices estratificados de acordo com a distribuição das labels.

    Parâmetros:
    - labels: array-like (lista ou np.array) com os rótulos das amostras
    - n_amostras: número total de amostras a serem selecionadas
    - random_state: semente para reprodução dos resultados

    Retorna:
    - lista de índices estratificados (shuffle final incluso)
    """
    labels = np.array(labels)
    unique_classes, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()

    # Quantidade de amostras por classe (inteiro)
    per_class_counts = (proportions * n_amostras).astype(int)

    # Corrige se a soma não der exatamente n_amostras
    while per_class_counts.sum() < n_amostras:
        remainder = (proportions * n_amostras) - per_class_counts
        idx = np.argmax(remainder)
        per_class_counts[idx] += 1

    # Seleciona os índices estratificados
    indices = []
    rng = np.random.default_rng(random_state)
    for cls, n in zip(unique_classes, per_class_counts):
        cls_indices = np.where(labels == cls)[0]
        selected = rng.choice(cls_indices, size=n, replace=False)
        indices.extend(selected)

    # Embaralha o resultado final
    indices = shuffle(indices, random_state=random_state)

    return indices
