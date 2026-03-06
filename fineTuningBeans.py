# ------------------------------------------------------------------------------------------------------------------
# ----------------------------- fine tune beans --------------------------------------------------------------------
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- Donnees 
train_path, test_path,  = 'imageAcquisition/train/', 'imageAcquisition/test/'
batch_size, img_rows, img_cols, num_class = 16, 224, 224, 10

train_datagen = ImageDataGenerator(validation_split = 0.3,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
# -----------------------------------------------------------------------------------------------------------------
train_generator = train_datagen.flow_from_directory(directory = train_path,
                                                    target_size = (img_rows, img_cols),
                                                    batch_size = batch_size,
                                                    class_mode = 'categorical',
                                                    color_mode = 'rgb',
                                                    shuffle=True)
# Vous pouvez ensuite afficher quelques images de la base :
x_batch, y_batch = train_generator.next()
fig = plt.figure()
columns = 4
rows = 4
for i in range(1, columns*rows):
    num = np.random.randint(batch_size)
    image = x_batch[num].astype(np.int)
    fig.add_subplot(rows, columns, i)
    plt.imshow(image)
    
plt.show()



