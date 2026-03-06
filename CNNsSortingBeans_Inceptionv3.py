"""---------------------------------------------------------------------------------------------------------
-------------------> Nous concevons dans cette partie l'architecture InceptionV3 des CNNs <-----------------
# -- << Il est constituee de 26 couches convolutives et 2 couches entièrement connectées >> ----------------
# -- << Nombre de params : 24 M environ et occupes plus de 96Mo de memoire >> ------------------------------
# -- << il est utiliser pour la classification -------------------------------------------------------------
# -- << profondeur de cette architecture 156 couches caches ------------------------------------------------
------------------------------------------------------------------------------------------------------------"""

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ mes bibliotheques @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3  # lecture du modele InceptionV3 de keras

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ initialisation des donnees @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
batch_size, num_class = 16, 10
img_rows, img_cols, img_chanel  = 224, 224, 3
# <>------<> image d'entree du CNNs ---------------------------------------------------------------------------
img_input = (img_rows, img_cols, img_chanel)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ creation de notre reseaux  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# <>------<> chargement du modele pre-entrainer Inception V3 depuis keras -------------------------------------
base_model = InceptionV3(include_top=False, weights='imagenet', pooling='avg', input_shape = img_input)
print(base_model.summary())  

# <<-D-E-B-U-T->> ANN avec InceptionV3 du type sequentielle ---------------------------------------------------
input_ann_model = base_model.input
ann_model = base_model.output
ann_model = Dense(1024, activation='relu')(ann_model)
ann_model = Dropout(0.60)(ann_model)
output_ann_model = Dense(num_class, activation='sigmoid')(ann_model)
# <<-F-I-N->> ANN ---------------------------------------------------------------------------------------------
# <>-----<> creation du model de notre CNNs InceptionV3 a entrainner ------------------------------------------
# <>-----<> les couches du CNN InceptionV3 sont figer ie pas de modification des poids ------------------------
for layer in base_model.layers:
    layer.trainable = False  
# <>-----<> modèle regroupe les couches dans un objet avec les caracteristique de formation et d'inference ---
model = Model(inputs=input_ann_model, outputs=output_ann_model)
print(model.summary())

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ entraiment de notre reseau de neurones @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# <>-----<> nous pouvons commencer avec le fine-Tuning a ce stade --------------------------------------------
train_path = 'imageAcquisition/train/'
train_datagen = ImageDataGenerator(validation_split = 0.3,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
# ------------------------------------------------------------------------------------------------------------
train_generator = train_datagen.flow_from_directory(directory = train_path,
                                                    target_size = (img_rows, img_cols),
                                                    batch_size = batch_size,
                                                    class_mode = 'categorical',
                                                    color_mode = 'rgb',
                                                    shuffle=True)
# <>-----<> verifier le nombre de parametre entrainable -----------------------------------------------------
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)
"""
# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True
"""
# <>-----<> debut de l'entrainement du model ---------------------------------------------------------------
opt = tf.keras.optimizers.SGD(lr = 0.01, momentum = 0.9)
model.compile(loss='categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              patience=5,
                              verbose=1,
                              factor=0.2,
                              min_delta=0.0001,
                              cooldown=1,
                              min_lr=0.0001)  
# <>-----<> Debut d'apprentissage ------------------------------------------------------------------------
history = model.fit_generator(train_generator, 
                              steps_per_epoch = train_generator.n/batch_size, 
                              epochs = 10, 
                              callbacks=[reduce_lr])
model.save('BeansTuning.h5') # summarize history for accuracy

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ courbe montrant l'erreur d'apprent @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
plt.plot(history.history['loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper left')
plt.show()


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ augmentation des donnees @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@