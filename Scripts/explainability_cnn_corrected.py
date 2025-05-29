import argparse
import os
import numpy as np
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
epoch = 15
    
# Argument Parsing\n",
def parse_arguments():
    parser = argparse.ArgumentParser(description="Explicability of CNN")
    parser.add_argument('train_env', type=str, help='training Scenario')   
    parser.add_argument('train_station', type=str, help='training station')
    parser.add_argument('test_env', type=str, help='testing Scenario')
    parser.add_argument('test_station', type=str, help='testing station')
    parser.add_argument('model_name', type=str, help='Name of the model file (.h5)')
    parser.add_argument('NoOfSubcarrier', type=int, help='No of Subcarrier')
    return parser.parse_args()
        
# Data Paths
# def get_data_paths(args):
#     train_env = args.train_env
#     train_station = args.train_station
#     test_env = args.test_env
#     test_station = args.test_station
    
#     Bw = "80MHz"
#     num_mon = "3mo"
#     slots = "Slots",
#     model_name = args.model_name
#     Train_dir = 'Train'
#     Test_dir = 'Test'
#     NoOfSubcarrier = int(args.NoOfSubcarrier)
    
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     data_path = "/scratch/diogo.alves/work/SiMWiSense/Data/fine_grained"
#     train_dir = os.path.join(data_path, args.train_env, Bw, num_mon, args.train_station, slots, Train_dir)
#     test_dir = os.path.join(data_path, args.test_env, Bw, num_mon, args.test_station, slots, Test_dir)
#     model_path = os.path.join("models", model_name)  # corrected to point to .h5 file"
                                      
#     return train_dir, test_dir, model_path, NoOfSubcarrier

def get_data_paths(args):
    Bw = "80MHz"
    num_mon = "3mo"
    slots = "Slots"
    Train_dir = "Train"
    Test_dir = "Test"
    NoOfSubcarrier = args.NoOfSubcarrier

    data_path = "/scratch/diogo.alves/work/SiMWiSense/Data/fine_grained"
    model_path = "/home/nobuko/SiMWiSense/models"
    train_dir = os.path.join(data_path, args.train_env, Bw, num_mon, args.train_station, slots, Train_dir)
    test_dir = os.path.join(data_path, args.test_env, Bw, num_mon, args.test_station, slots, Test_dir)
    #model_dir = os.path.join(data_path, args.train_env, Bw, num_mon, args.train_station, args.model_name)
    model_dir = os.path.join(model_path, args.model_name)
    print(model_dir)

    return train_dir, test_dir, model_dir, NoOfSubcarrier

    
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
    print("DEBUG: train_env:", args.train_env, type(args.train_env))
    print("DEBUG: train_station:", args.train_station, type(args.train_station))
    print("DEBUG: test_env:", args.test_env, type(args.test_env))
    print("DEBUG: test_station:", args.test_station, type(args.test_station))
    print("DEBUG: model_name:", args.model_name, type(args.model_name))
    print("DEBUG: NoOfSubcarrier:", args.NoOfSubcarrier, type(args.NoOfSubcarrier))

    train_dir, test_dir, model_path, NoOfSubcarrier = get_data_paths(args)
    labels = list("ABCDEFGHIJKLMNOPQRST")

    test_csv = os.path.join(test_dir, 'test_set.csv')

    test_gen = DataGenerator(test_dir, test_csv, NoOfSubcarrier, len(labels), (window_size, NoOfSubcarrier, 2), batchsize=64, shuffle=False)

    print("Test gen:",len(test_gen))
    
    model = load_model(model_path)
    
    figures_dir = "/home/nobuko/SiMWiSense/figures"
    
    #final_loss, final_accuracy = model.evaluate(test_gen)
    #print('Test Loss: {}, Test Accuracy: {}'.format(final_loss, final_accuracy))

    #evaluate_and_plot_confusion_matrix(model, test_gen, labels, figures_dir)

    print("[SHAP] Gerando explicação com SHAP...")

    # Testando juntar vários samples
    X_samples, _ = test_gen.__getitem__(0)

    X_samplestmp1, _ =  test_gen.__getitem__(1)

    X_samples = np.concatenate([X_samples,X_samplestmp1],0)

    X_samplestmp2, _ =  test_gen.__getitem__(2)

    X_samples = np.concatenate([X_samples,X_samplestmp2],0)

    X_samplestmp3, _ =  test_gen.__getitem__(3)

    X_samples = np.concatenate([X_samples,X_samplestmp3],0)

    print("Xsamples shape:", X_samples.shape)
    
    # Tentativa 1 - Não funcionou
    #explainer = shap.Explainer(model, X_samples)
    #print("[SHAP] Criou X_samples e explainer")
    #print("Type X_samples",type(X_samples))
    #print("Dim X_samples",X_samples.shape)
    #shap_values = explainer(X_samples[:1])

    background = X_samples[:100]  # SHAP needs a background dataset
     
    explainer = shap.GradientExplainer(model, background)
    print("[SHAP] criou o explainer")
    
    shap_values = explainer(X_samples[:1]) 
    print("[SHAP] criou os valores shap")
    print("Type shap_values",type(shap_values))
    print("shap shap_values",shap_values.shape)

    #print(shap_values)
    np.savez_compressed(
        "shap_output_sample1.npz",
        values=shap_values.values,
        data=shap_values.data,
        base_values=shap_values.base_values
    )
    
    #shap.plots.bar(shap_values[0])
    shap_img = shap_values[0].values[0]  # shape should be (50, 242, 2)

    # Aggregate across the "channel" dimension (if needed)
    shap_magnitude = np.abs(shap_img).sum(axis=-1)  # shape (50, 242)

    # Keep only the top 5%
    threshold = np.percentile(shap_magnitude, 95)  
    significant_mask = shap_magnitude >= threshold

    plt.figure(figsize=(12, 6))
    sns.heatmap(shap_magnitude, cmap='viridis')
    plt.title("SHAP Value Magnitude (Time x Subcarriers)")
    plt.xlabel("Subcarriers")
    plt.ylabel("Time")
    plt.savefig(os.path.join(figures_dir, 'testeShapTop5.png'), dpi=300)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title("SHAP values (magnitude) for input sample 0")
    plt.imshow(shap_magnitude, cmap='viridis', aspect='auto')
    plt.colorbar(label='SHAP Value Magnitude')
    plt.ylabel("Subcarriers")
    plt.xlabel("Real / Imaginary")
    plt.savefig(os.path.join(figures_dir, 'testeShap.png'), dpi=300)
    plt.show()

    print("[t-SNE] Reduzindo dimensão da penúltima camada...")
    penultimate_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    X_feats = penultimate_model.predict(test_gen)
    tsne = TSNE(n_components=2, random_state=42)
    Z = tsne.fit_transform(X_feats)
    y_true = np.argmax(test_gen.labels[test_gen.indexes], axis=1)[:len(Z)]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=y_true, palette='tab20', s=60)
    plt.title("t-SNE das ativações da penúltima camada")
    plt.show()

    print("[LIME] Aplicando LIME sobre uma amostra...")
    sample = X_samples[5]
    mag = np.sqrt(sample[:, :, 0]**2 + sample[:, :, 1]**2)
    rgb_sample = np.stack([mag] * 3, axis=2)

    def predict_fn(imgs):
        imgs = np.stack([img[:, :, 0] for img in imgs])
        imgs = np.expand_dims(imgs, -1)
        imgs = np.array([np.transpose(img, (1, 0, 2)) for img in imgs])
        return model.predict(imgs)

    lime_exp = lime_image.LimeImageExplainer()
    explanation = lime_exp.explain_instance(rgb_sample.astype('double'), predict_fn, top_labels=1, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(label=explanation.top_labels[0], positive_only=True, hide_rest=False)

    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title("LIME: regiões relevantes no CSI")
    plt.axis("off")
    plt.show()
    
if __name__ == '__main__':
    main()
