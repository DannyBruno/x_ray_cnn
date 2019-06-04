"""

Load dataset from train/val/test directories.

"""

import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import imgaug.augmenters as iaa

from model import build_model

from pathlib import Path
import cv2

# Define base paths.
data_path = Path('./data')
train_path = data_path / 'train'
val_path = data_path / 'val'
test_path = data_path / 'test'

def create_df(null_case_path, positive_case_path):

    # Grab file names.
    null_cases = null_case_path.glob('*.jpeg')
    positive_cases = positive_case_path.glob('*.jpeg')

    data = []

    for image_path in null_cases:
        data.append((image_path, 0))

    for image_path in positive_cases:
        data.append((image_path, 1))

    # Convert to df.
    return pd.DataFrame(data, columns=['image_path', 'label'], index=None)



def load_images(df):
    '''
    Convert images to 3-channel, normalize pixels, resize to 224x224.
    '''

    validated_data = []
    data_labels = []

    # Select all normal cases.
    for idx, row in df.iterrows():
        # Load.
        img_path = row['image_path']
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (224, 224))   # Resize.
        if img.shape[2] == 1:       # 3 channel.
            img = np.dstack([img, img, img])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255     # Normalize.

        # Add to arr.
        validated_data.append(img)
        data_labels.append(to_categorical(row['label'], num_classes=2))


    return np.array(validated_data), np.array(data_labels)


def augmentation_generator(data, batch_size):
    '''
    Performs data augmentation on training set. Practice data gen.
    '''
    n = len(data)
    batches_to_make = n//batch_size

    # Containers for batch data and labels.
    batch_data = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size, 2), dtype=np.float32)

    # Grab indices.
    indices = np.arange(n)

    # Define transformations.
    transformations = iaa.OneOf([
        iaa.Fliplr(0.5),    # Flip.
        iaa.ContrastNormalization((0.75, 1.5)),  # Play around with contrast.
        iaa.GaussianBlur(sigma=(0, 0.5)),   # Randomly blur.
        iaa.Multiply((0.8, 1.2))    # Change brightness.
    ])

    # Start a counter.
    batches_made = 0
    while True:

        # Make batch.
        np.random.shuffle(indices)
        count = 0
        next_batch = indices[(batches_made*batch_size):(batches_made+1)*batch_size]
        for foo, idx in enumerate(next_batch):
            img_name = data.iloc[idx]['image_path']
            img_label = data.iloc[idx]['label']

            # Encode label.
            encoded_label = to_categorical(img_label, num_classes=2)

            # Read image and resize.
            img = cv2.imread(str(img_name))
            img = cv2.resize(img, (224, 224))

            # Check for greyscale.
            if img.shape[2] == 1:
                img = np.dstack([img, img, img])

            # Change from BGR to RGB and normalize.
            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_img = orig_img.astype(np.float32)/255

            batch_data[count] = orig_img
            batch_labels[count] = encoded_label

            # Data augment for under-sampled class.
            if img_label == 0 and count < batch_size - 2:
                # Image 1.
                aug_img1 = transformations.augment(images=img)
                aug_img1 = cv2.cvtColor(aug_img1, cv2.COLOR_BGR2RGB)
                aug_img1 = aug_img1.astype(np.float32)/255
                batch_data[count + 1] = aug_img1
                batch_labels[count + 1] = encoded_label

                # Image 2.
                aug_img2 = transformations.augment(images=img)
                aug_img2 = cv2.cvtColor(aug_img2, cv2.COLOR_BGR2RGB)
                aug_img2 = aug_img2.astype(np.float32)/255
                batch_data[count + 2] = aug_img2
                batch_labels[count + 2] = encoded_label

            else:
                count += 1

            # Can't add anymore.
            if count == batch_size - 1:
                break


        batches_made += 1
        yield batch_data, batch_labels


        # Reset for the next go.
        if batches_made >= batches_to_make:
            batches_made = 0


if __name__ == '__main__':

    '''
    Prepare data.
    '''
    train_df = create_df(train_path / 'NORMAL', train_path / 'PNEUMONIA')
    val_df = create_df(val_path / 'NORMAL', val_path / 'PNEUMONIA')
    test_df = create_df(test_path / 'NORMAL', test_path / 'PNEUMONIA')
    # Observe things about the data...
    print(train_df['label'].value_counts())    # A lot less normal cases than pneumonia. Augment normal cases.

    # Load test and validation sets.
    val_imgs, val_labels = load_images(val_df)
    test_imgs, test_labels = load_images(test_df)

    # Create data generator out of training data.
    train_data_generator = augmentation_generator(train_df, batch_size=16)

    '''
    Build & train model.
    '''
    model = build_model()
    model.summary()

    opt = Adam(lr=.0001, decay=1e-5)
    es = EarlyStopping(patience=5)
    chkpt = ModelCheckpoint(filepath='best_model_to_date', save_best_only=True, save_weights_only=True)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)

    batch_size = 16
    nb_epochs = 20

    history = model.fit_generator(train_data_generator, epochs=nb_epochs, steps_per_epoch=train_df.shape[0]//batch_size,
                                  validation_data=(val_imgs, val_labels), callbacks=[es, chkpt],
                                  class_weight={0: 1.0, 1: 0.4})

    test_loss, test_score = model.evaluate(test_imgs, test_labels)
    print(f'Test_loss: {test_loss}, Test_score: {test_score}')




