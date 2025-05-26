# Trained Models

This folder contains fine-grained trained models, used in the explainability experiments, saved in HDF5 (`.h5`) format. Each model was trained on different subsets of data collected from various environments/monitors. The naming convention indicates the training environment, monitor, and the number of subcarriers used.

## Model Naming Convention

fine_grained_trainedon_\<Environment\>_\<Monitor\>_242.h5

- `<Environment>`: The environment the model was trained on (`Classroom`, `Office`, or `ClassroomOffice` for combined environments).
- `<Monitor>`: Indicates the specific monitor (`m1`, `m2`, `m3`, or combinations like `m1m2m3`).
- `242`: Refers to the number of subcarriers used in the model input (i.e., 242 subcarriers of CSI data).

## List of Models

- `fine_grained_trainedon_Classroom_m1_242.h5`
- `fine_grained_trainedon_Classroom_m2_242.h5`
- `fine_grained_trainedon_Classroom_m3_242.h5`
- `fine_grained_trainedon_ClassroomOffice_m1m2m3_242.h5`
- `fine_grained_trainedon_Office_m1_242.h5`
- `fine_grained_trainedon_Office_m2_242.h5`
- `fine_grained_trainedon_Office_m3_242.h5`

Each file represents a trained model that can be loaded using Keras or TensorFlow for inference or further fine-tuning.

## Usage

To load a model:

```python
from tensorflow.keras.models import load_model

model = load_model('trained_models/fine_grained_trainedon_Classroom_m1_242.h5')

