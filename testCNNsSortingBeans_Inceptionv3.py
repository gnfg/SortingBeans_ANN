"""---------------------------------------------------------------------------------------------------------
------------------------> TEST DE NOTRE ANNs pour la classifications des graines <--------------------------
# -- << Il est constituee de 26 couches convolutives et 2 couches entièrement connectées >> ----------------
# -- << Nombre de params : 24 M environ et occupes plus de 96Mo de memoire >> ------------------------------
# -- << il est utiliser pour la classification -------------------------------------------------------------
# -- << profondeur de cette architecture 156 couches caches ------------------------------------------------
------------------------------------------------------------------------------------------------------------"""
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ initialisation des donnees @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
batch_size, num_class = 16, 10
img_rows, img_cols  = 224, 224
# <>------<> image d'entree du CNNs ---------------------------------------------------------------------------
img_input = (img_rows, img_cols)

test_path = 'imageAcquisition/test/'
model = load_model('BeansTuning.h5')
test_datagen = ImageDataGenerator()
# -------------------------------------------------------------------------------------------------------------
test_generator = test_datagen.flow_from_directory(directory = test_path,
                                                  target_size = img_input,
                                                  color_mode = 'rgb',
                                                  shuffle = False,
                                                  class_mode = 'categorical',
                                                  batch_size = batch_size)  # 1
# -------------------------------------------------------------------------------------------------------------
filenames = test_generator.filenames
nb_samples = len(filenames)
fig=plt.figure()
rows, columns = 4, 4
for i in range(1, columns*rows -1):
    x_batch, y_batch = test_generator.next()
    name = model.predict(x_batch)
    name = np.argmax(name, axis=-1)
    true_name = y_batch
    true_name = np.argmax(true_name, axis=-1)
    label_map = (test_generator.class_indices)
    label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
    predictions = [label_map[k] for k in name]
    true_value = [label_map[k] for k in true_name]
    image = x_batch[0].astype(np.int)
    fig.add_subplot(rows, columns, i)
    plt.title(str(predictions[0]) + ':' + str(true_value[0]))
    plt.imshow(image)
plt.show()
