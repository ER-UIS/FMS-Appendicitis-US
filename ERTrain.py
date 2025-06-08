from KerasModels.inception_resnet_v2  import InceptionResNetV2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.optimizers import Adam
import keras
import os
import time
import json
import warnings
from keras.callbacks import Callback
class EarlyStoppingAtAccuracy(Callback):
    def __init__(self, target_accuracy=1.0):
        super(EarlyStoppingAtAccuracy, self).__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= self.target_accuracy:
            print(f"\nReached {self.target_accuracy * 100:.2f}% accuracy, stopping training!")
            self.model.stop_training = True
execution_path = os.getcwd()
initial_learning_rate = 0.01 

def lr_schedule(epoch):
    """
    Learning Rate Schedule
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 1e-4
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1

    print('Learning rate: ', lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

def ERtrainModel(Data_Path):
    global DATASET_DIR, DATASET_TRAIN_DIR, DATASET_TEST_DIR, DATASET_Json_DIR, DATASET_Logs_DIR, DATASET_MODELs_DIR, DATASET_TestModel_DIR, DATASET_BestModel_DIR
            
   
    DATASET_DIR = os.path.join(execution_path, Data_Path)  # main dataset directory
    DATASET_TRAIN_DIR = os.path.join(DATASET_DIR, "train")  # training data directory
    DATASET_TEST_DIR = os.path.join(DATASET_DIR, "test")  # testing data directory
    DATASET_Json_DIR = os.path.join(DATASET_DIR, "json")  # JSON storage directory
    DATASET_Logs_DIR = os.path.join(DATASET_DIR, "logs")  # logs storage directory
    DATASET_MODELs_DIR = os.path.join(DATASET_DIR, "models")  # models storage directory
    DATASET_TestModel_DIR = os.path.join(DATASET_DIR, "TestModel")  # test model storage directory
    DATASET_BestModel_DIR = os.path.join(DATASET_DIR, "BestModel")  # best model storage directory
    dir_time = time.strftime('%Y%m%d', time.localtime(time.time()))  # current date directory
    DATASET_TestModel_timeDIR = os.path.join(DATASET_TestModel_DIR, dir_time)  # timestamped test model directory
    directories = [DATASET_DIR, DATASET_TRAIN_DIR, DATASET_TEST_DIR, DATASET_Json_DIR, DATASET_Logs_DIR, DATASET_MODELs_DIR, DATASET_TestModel_DIR, DATASET_BestModel_DIR, DATASET_TestModel_timeDIR]

    for directory in directories:
        if not os.path.isdir(directory):  # create directory if it does not exist
            os.makedirs(directory)
    
def trainModel(modelType, num_objects, num_experiments, enhance_data=False, batch_size=32, 
               initial_learning_rate=1e-3, show_network_summary=False, training_image_size=224, 
               continue_from_model=None, transfer_from_model=None, transfer_with_full_training=None, 
               initial_num_objects=None, save_full_model=False):
    
    num_classes = num_objects
    num_epochs = num_experiments
    m_type = None
    base_Path = 'model_appendix' 
    if isinstance(modelType, type(InceptionResNetV2)):
        m_type = "InceptionResNetV2"
    
    if training_image_size < 100:
        warnings.warn(f"The specified training_image_size {training_image_size} is less than 100. Hence the training_image_size will default to 100.")
        training_image_size = 100

    model_path = os.path.join(base_Path, "InceptionResNetV2-ERmodel.h5")

    if isinstance(modelType, type(InceptionResNetV2)):
        if continue_from_model:
            model = InceptionResNetV2(input_shape=(training_image_size, training_image_size, 3), 
                             weights=model_path, classes=num_objects, include_top=True)
            
            if show_network_summary:
                print("Training using weights from a previously saved model")
        elif transfer_from_model:
            base_model =InceptionResNetV2(input_shape=(training_image_size, training_image_size, 3), 
                                  weights=model_path, include_top=False, pooling="avg")

            network = base_model.output
            network = Dense(num_objects, activation='softmax', use_bias=True)(network)
            
            model = keras.models.Model(inputs=base_model.input, outputs=network)

            if show_network_summary:
                print("Training using weights from a pre-trained InceptionResNetV2 model")
        else:
            base_model =InceptionResNetV2(input_shape=(training_image_size, training_image_size, 3), 
                                  weights=None, classes=num_classes, include_top=False, pooling="avg")

            network = base_model.output
            network = Dense(num_objects, activation='softmax', use_bias=True)(network)
            
            model = keras.models.Model(inputs=base_model.input, outputs=network)
    
    optimizer = Adam(lr=initial_learning_rate, decay=1e-4)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    if show_network_summary:
        model.summary()

    save_weights_condition = not save_full_model
    
    if m_type == "InceptionResNetV2":
        if continue_from_model:
            model_name = 'InceptionResNetV2-continue-ERmodel_ex-{epoch:03d}_acc-{accuracy:03f}.h5'
        elif transfer_from_model:
            model_name = 'InceptionResNetV2-imagenet-transfer-ERmodel_ex-{epoch:03d}_acc-{accuracy:03f}.h5'
        else:
            model_name = 'InceptionResNetV2-ERmodel_ex-{epoch:03d}_acc-{accuracy:03f}.h5'
    
    model_path = os.path.join(DATASET_MODELs_DIR, model_name)

    checkpoint = ModelCheckpoint(filepath=model_path,
                                 monitor='accuracy',
                                 verbose=1,
                                 save_weights_only=save_weights_condition,
                                 save_best_only=True,
                                 period=1)

    log_name = 'lr-{}_{}'.format(initial_learning_rate, time.strftime("%Y-%m-%d-%H-%M-%S"))
    logs_path = os.path.join(DATASET_Logs_DIR, log_name)

    tensorboard = TensorBoard(log_dir=logs_path, 
                              histogram_freq=0, 
                              write_graph=False, 
                              write_images=False)
    
    if enhance_data:
        print("Using Enhanced Data Generation")

    height_shift = 0.1 if enhance_data else 0
    width_shift = 0.1 if enhance_data else 0

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=enhance_data,
        height_shift_range=height_shift,
        width_shift_range=width_shift
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    train_generator = train_datagen.flow_from_directory(DATASET_TRAIN_DIR, 
                                                        target_size=(training_image_size, training_image_size),
                                                        batch_size=batch_size,
                                                        class_mode="categorical")

    test_generator = test_datagen.flow_from_directory(DATASET_TEST_DIR, 
                                                      target_size=(training_image_size, training_image_size),
                                                      batch_size=batch_size,
                                                      class_mode="categorical")

    class_indices = train_generator.class_indices
    class_json = {str(class_indices[eachClass]): eachClass for eachClass in class_indices}

    with open(os.path.join(DATASET_Json_DIR, "model_class.json"), "w+") as json_file:
        json.dump(class_json, json_file, indent=4, separators=(",", " : "), ensure_ascii=True)
    
    print("JSON Mapping for the model classes saved to ", os.path.join(DATASET_Json_DIR, "model_class.json"))

    num_train = len(train_generator.filenames)
    num_test = len(test_generator.filenames)
    print("Number of experiments (Epochs) : ", num_epochs)

    early_stopping = EarlyStoppingAtAccuracy(target_accuracy=1.0)

    model.fit_generator(train_generator, steps_per_epoch=int(num_train / batch_size), epochs=num_epochs,
                            validation_data=test_generator,
                            validation_steps=int(num_test / batch_size), callbacks=[checkpoint,tensorboard,lr_scheduler, early_stopping])
      

