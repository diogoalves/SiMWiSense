import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from dataGenerator import DataGenerator_Coarse, DataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import shap
from lime import lime_image
from sklearn.manifold import TSNE
from skimage.segmentation import mark_boundaries
    
    
# Constants\n",
num_classes = 20
window_size = 50

    
# Argument Parsing\n",
def parse_arguments():
    parser = argparse.ArgumentParser(description="Explicability of CNN")  
    parser.add_argument('test_env', type=str, help='testing Scenario')
    parser.add_argument('test_station', type=str, help='testing station')
    parser.add_argument('model_name', type=str, help='Name of the model file (.h5)')
    parser.add_argument('NoOfSubcarrier', type=int, help='No of Subcarrier')
    return parser.parse_args()
        

def get_data_paths(args):
    Bw = "80MHz"
    num_mon = "3mo"
    slots = "Slots"
    Test_dir = "Test"
    NoOfSubcarrier = args.NoOfSubcarrier

    data_path = "/scratch/diogo.alves/work/SiMWiSense/Data/fine_grained"
    model_path = "/home/nobuko/SiMWiSense/models"
    
    test_dir = os.path.join(data_path, args.test_env, Bw, num_mon, args.test_station, slots, Test_dir)
    print(test_dir)
    
    model_dir = os.path.join(model_path, args.model_name)
    print(model_dir)

    return test_dir, model_dir, NoOfSubcarrier

    
def evaluate_and_plot_confusion_matrix(model, test_gen, labels, figures_dir):
    Y = test_gen.labels[test_gen.indexes]
    Y_true = np.zeros(len(Y))
    for i, e in enumerate(Y):
        Y_true[i] = labels.index(e)
    
    Y_pred = model.predict(test_gen)
    Y_pred = np.argmax(Y_pred, axis=1)
    
    cm = confusion_matrix(Y_true[:len(Y_pred)], Y_pred, normalize='true')
    plt.figure(figsize=(32, 32))
    ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, fmt='.1f', square=True, xticklabels=labels, yticklabels=labels)
    ax.set_ylabel('Actual', fontsize=20)
    ax.set_xlabel('Predicted', fontsize=20)
    plt.savefig(os.path.join(figures_dir, 'confusion_matrix.png'), dpi=300)
    plt.show()

def main():
    args = parse_arguments()
    print("DEBUG: test_env:", args.test_env, type(args.test_env))
    print("DEBUG: test_station:", args.test_station, type(args.test_station))
    print("DEBUG: model_name:", args.model_name, type(args.model_name))
    print("DEBUG: NoOfSubcarrier:", args.NoOfSubcarrier, type(args.NoOfSubcarrier))

    test_env = args.test_env
    test_station = args.test_station
    
    test_dir, model_path, NoOfSubcarrier = get_data_paths(args)
    labels = list("ABCDEFGHIJKLMNOPQRST")

    #fine_grained set: test filtred csv

    test_csv = os.path.join(test_dir, 'test_set.csv')
    df_test_csv = pd.read_csv(test_csv, header=None)
    test_csv_len = len(df_test_csv)
    print("Test cs:",test_csv_len)

    test_gen = DataGenerator(test_dir, test_csv, NoOfSubcarrier, len(labels), (window_size, NoOfSubcarrier, 2), batchsize=64, shuffle=False)

    test_gen_len = len(test_gen) 
    
    print("Test gen length:",test_gen_len)
    
    model = load_model(model_path)
    
    #figures_dir = "/home/nobuko/SiMWiSense/figuresOffice"
    figures_dir = "/home/nobuko/SiMWiSense/figuresClassroom"

    print("[SHAP] Gerando explicação com SHAP...")

    # Testando juntar vários batches para ter um X_samples maior 
    # para fazer o background.
    
    rng = np.random.default_rng()
    
    samples_ids = rng.integers(low=1, high=test_gen_len, size=20)

    print("Generated samples",samples_ids)
    
    Xlabels = []
    X_samples, Xlabels = test_gen.__getitem__(samples_ids[0])
    for i in range(2,len(samples_ids)):
        samplestmp, tmpLabels =  test_gen.__getitem__(samples_ids[i])
        X_samples = np.concatenate([X_samples,samplestmp],0)
        Xlabels = np.concatenate((Xlabels,tmpLabels),0)

    print("Xsamples shape:", X_samples.shape)

    print("Xsamples Labels ",Xlabels)

    print("Xlabels shape",Xlabels.shape)
    
    background = X_samples[:500]  # SHAP needs a background dataset
     
    explainer = shap.GradientExplainer(model, background)
    print("[SHAP] criou o explainer e vai chamá-lo com os exemplos")

    # Para gerar para todas as matrizes
    #for currentMat in range(test_gen_len):

    Xlabels_len = len(Xlabels) 
    
    samples_img_ids = rng.integers(low=1, high=Xlabels_len, size=50)

    print("Generated samples",samples_img_ids)
    
    # Para gerar para algumas imagens
    for currentMat in samples_img_ids:
        shap_values = explainer(X_samples[currentMat:currentMat+1])
        #print("[SHAP] criou os valores e vai verificar a predição para esse exemplo")
    
    # ====== Identifica a classe prevista pelo modelo ======

        #print("Entrada do modelo",model.inputs)
        #print("Entrada dada",type(X_samples[currentMat:currentMat+1]))
        #print("Shape da entrada dada",X_samples[currentMat:currentMat+1].shape)
    
        preds = model.predict(X_samples[currentMat:currentMat+1])  # Shape (1, 20)
        #print(preds)
        predicted_class = np.argmax(preds[0])  # Índice da classe prevista
        #print(f"[SHAP] Classe prevista para a amostra: {predicted_class} ({labels[predicted_class]})")

        # Print the ground truth

        groundtruth_class = np.argmax(Xlabels[currentMat:currentMat+1])
        #print(f"[SHAP] Classe groundtruth para a amostra: {groundtruth_class} ({labels[groundtruth_class]})")

        # Verifica se shap_values é uma lista (SHAP retorna uma lista com uma entrada por classe)
        #print(f"[SHAP] Número de classes explicadas: {len(shap_values)}")
        #print(f"[SHAP] Shape do shap_values: {shap_values.shape}")
    
        # Acessa os SHAP values da classe prevista
        shap_val = shap_values.values[0, :, :, :, predicted_class]  # shape (50, 242, 2)
    
        #print("[SHAP] criou os valores shap")
        #print("Type shap_values",type(shap_values))
        #print("shap shap_values",shap_values.shape)

        #print(shap_values)
        #    np.savez_compressed(
        #        "shap_output_sample1.npz",
        #        values=shap_values.values,
        #        data=shap_values.data,
        #       6 base_values=shap_values.base_values
        #    )

        # Aggregate across the "channel" dimension (if needed)
        shap_magnitude = np.abs(shap_val).sum(axis=-1)  # shape (50, 242)

        plt.figure(figsize=(12, 6))
        sns.heatmap(shap_magnitude, cmap='viridis')
        plt.title(f"{test_env} - {test_station} SHAP Magnitude — Image {currentMat}, Pred {labels[predicted_class]}, GT {labels[groundtruth_class]}")
        plt.xlabel("Subcarriers")
        plt.ylabel("Time")
        fileName = str(test_env) + "_" + str(test_station) + "_" + "Shap_" + str(currentMat) + ".png"
        plt.savefig(os.path.join(figures_dir, fileName), dpi=50)
        plt.close()
        #plt.show()

        # Keep only the top 5%
        threshold = np.percentile(shap_magnitude, 95)  
        significantMask = shap_magnitude >= threshold

        plt.figure(figsize=(12, 6))
        plt.title(f"{test_env} - {test_station} Top5% Magnitude — Image {currentMat}, Pred {labels[predicted_class]}, GT {labels[groundtruth_class]}")
        plt.imshow(significantMask, cmap='viridis', aspect='auto')
        plt.colorbar(label='Top 5% SHAP Value Magnitude')
        plt.xlabel("Subcarriers")
        plt.ylabel("Time")
        fileNameTop5 = str(test_env) + "_" + str(test_station) + "_" + "ShapTop5_" + str(currentMat) + ".png"
        plt.savefig(os.path.join(figures_dir, fileNameTop5), dpi=50)
        plt.close()
        #plt.show()



    #print("[LIME] Aplicando LIME sobre uma amostra...")
    #sample = X_samples[5]
    #mag = np.sqrt(sample[:, :, 0]**2 + sample[:, :, 1]**2)
    #rgb_sample = np.stack([mag] * 3, axis=2)

    #def predict_fn(imgs):
    #    imgs = np.stack([img[:, :, 0] for img in imgs])
    #    imgs = np.expand_dims(imgs, -1)
    #    imgs = np.array([np.transpose(img, (1, 0, 2)) for img in imgs])
    #    return model.predict(imgs)

    #lime_exp = lime_image.LimeImageExplainer()
    #explanation = lime_exp.explain_instance(rgb_sample.astype('double'), predict_fn, top_labels=1, num_samples=1000)
    #temp, mask = explanation.get_image_and_mask(label=explanation.top_labels[0], positive_only=True, hide_rest=False)

    #plt.imshow(mark_boundaries(temp / 255.0, mask))
    #plt.title("LIME: regiões relevantes no CSI")
    #plt.axis("off")
    #plt.show()
    
if __name__ == '__main__':
    main()
