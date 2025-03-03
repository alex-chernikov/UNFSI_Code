import numpy as np
import tensorflow as tf
import const

tf.get_logger().setLevel('ERROR')


def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.1)
    image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.1)
    return image


def load_and_preprocess_image_MobNetV2(image,res=224):
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [res, res])
    image = image / 255.0
    return image


def load_and_preprocess_augment_image_MobNetV2(path):
    image=load_and_preprocess_image_MobNetV2(path)
    image = augment(image)
    return image


def get_callbacks(weights_path):
    from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',  # Monitor validation accuracy
        factor=0.66,  # Factor by which the learning rate will be reduced
        patience=150,  # Number of epochs with no improvement after which learning rate will be reduced
        verbose=1,  # Verbosity mode, 1 = update messages
        mode='min',  # Mode for the metric, 'max' to maximize val_accuracy
        min_lr=1e-6  # Lower bound on the learning rate
    )

    # Save the model weights when validation accuracy improves
    checkpoint = ModelCheckpoint(
        filepath=weights_path,  # File path to save the model
        monitor='val_loss',  # Monitor validation accuracy
        save_best_only=True,  # Only save the model when val_accuracy improves
        save_weights_only=True,
        mode='min',  # Mode for the metric, 'max' to maximize val_accuracy
        verbose=1  # Verbosity mode, 1 = update messages
    )

    # List of callbacks to pass to the model during training
    callbacks_list = [reduce_lr,checkpoint]
    return callbacks_list


def train_cls(mod_name,train_paths,train_labels,val_paths, val_labels, batch_size=32,ep=100):
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV3Small
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Dropout
    from tensorflow.keras.models import Model
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    from tensorflow.keras.losses import CategoricalCrossentropy

    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))

    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    encoder = label_encoder.fit(train_labels)
    encoded_labels_train=encoder.transform(train_labels)
    encoded_labels_val=encoder.transform(val_labels)

    # Convert labels to one-hot encoding
    num_classes = len(np.unique(encoded_labels_train))
    train_labels = tf.keras.utils.to_categorical(encoded_labels_train, num_classes=num_classes)
    val_labels = tf.keras.utils.to_categorical(encoded_labels_val, num_classes=num_classes)

    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_dataset = train_dataset.map(lambda x, y: (load_and_preprocess_augment_image_MobNetV2(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_dataset = val_dataset.map(lambda x, y: (load_and_preprocess_image_MobNetV2(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and shuffle the datasets
    train_dataset = train_dataset.shuffle(buffer_size=len(train_paths)).batch(batch_size).prefetch(
        buffer_size=len(train_paths)*2)
    val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=len(val_paths))

    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    optimizer='sgd'
    act_fun='relu'

    loss=CategoricalCrossentropy(label_smoothing=0.1)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x=Dropout(0.75)(x) # 84.70
    x = Dense(64, activation=act_fun)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    print(model.summary())

    mod_type='mobilenetv3'
    mod_name='fv5Tr_75Dr_Smooth_NoPad'

    save_path=const.MODELS_PATH + f'Classification/unf_{mod_type}_{mod_name}_custom_{optimizer}_{act_fun}_b{batch_size}_{ep}ep.keras'
    save_weights_path=const.MODELS_PATH+f'Classification/unf_{mod_type}_{mod_name}_custom_{optimizer}_{act_fun}_b{batch_size}_{ep}ep.weights.h5'
    callbacks=get_callbacks(save_weights_path)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    model.fit(train_dataset,
              validation_data=val_dataset,
              epochs=ep,
              callbacks=callbacks)
    model.save(save_path)

    loss, accuracy = model.evaluate(val_dataset)
    print(f'Validation accuracy: {accuracy}')


def evaluate_cls(model_path, paths, labels, weights=None,batch_size=512, plot=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report,accuracy_score
    from os.path import basename

    classes=list(const.CLASSES.values())

    val_dataset = tf.data.Dataset.from_tensor_slices((paths[:], labels[:]))
    val_dataset = val_dataset.map(lambda x, y: (load_and_preprocess_image_MobNetV2(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=10*batch_size)

    model = tf.keras.models.load_model(model_path)
    if weights:
        model.load_weights(weights)
    preds=model.predict(val_dataset)
    preds=np.argmax(preds, axis=1)

    print(classification_report(labels, preds))
    acc=accuracy_score(labels, preds)
    print(basename(model_path),acc)

    if plot:
        cm = tf.math.confusion_matrix(labels, preds)
        cm=cm.cpu().numpy().T
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes,cbar=False, ax=ax, annot_kws={"size": 16})
        ax.set_title(None)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.ylabel('Predicted',fontsize=12)
        plt.xlabel('True',fontsize=12)
        plt.show()