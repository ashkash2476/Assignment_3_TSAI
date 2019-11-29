# Assignment_3_TSAI

## Final Validation accuracy for Base Network
Epoch 50/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3400 - acc: 0.8876 - val_loss: 0.6203 - val_acc: 0.8123


## Model definition (model.add... ) with output channel size and receptive field


model_2 = Sequential()

model_2.add(SeparableConv2D(32, kernel_size=(3, 3), depth_multiplier=2,input_shape=(32, 32, 3))) # Output Size: 30,30,32   RF: 3
model_2.add(BatchNormalization())
model_2.add(Activation('relu'))
model_2.add(Dropout(0.15))

model_2.add(SeparableConv2D(64, kernel_size=(3, 3), border_mode = 'same'))  # 28,28,64 RF: 5
model_2.add(BatchNormalization())
model_2.add(Activation('relu'))
model_2.add(Dropout(0.15))


model_2.add(SeparableConv2D(64, kernel_size=(3, 3)))  # 28,28,64 RF: 7
model_2.add(BatchNormalization())
model_2.add(Activation('relu'))
model_2.add(Dropout(0.15))

model_2.add(SeparableConv2D(128, kernel_size=(3, 3), border_mode = 'same'))  # 26,26,128  RF:8
model_2.add(BatchNormalization())
model_2.add(Activation('relu'))
model_2.add(Dropout(0.15))


model_2.add(SeparableConv2D(128, kernel_size=(3, 3)))  # 24,24,128 RF:12
model_2.add(BatchNormalization())
model_2.add(Activation('relu'))
model_2.add(Dropout(0.15))

model_2.add(MaxPooling2D(pool_size=(2, 2))) # 12,12,128  RF:13

model_2.add(SeparableConv2D(32, kernel_size=(3, 3), border_mode = 'same'))  # 10,10,32   RF: 17
model_2.add(BatchNormalization())
model_2.add(Activation('relu'))
model_2.add(Dropout(0.15))


model_2.add(SeparableConv2D(64, kernel_size=(3, 3)))  # 28,28,64 RF: 21
model_2.add(BatchNormalization())
model_2.add(Activation('relu'))
model_2.add(Dropout(0.15))

model_2.add(SeparableConv2D(128, kernel_size=(3, 3), border_mode = 'same'))  # 26,26,128  RF:25
model_2.add(BatchNormalization())
model_2.add(Activation('relu'))
model_2.add(Dropout(0.15))


model_2.add(SeparableConv2D(128, kernel_size=(3, 3), border_mode = 'same'))  # 26,26,128  RF:29
model_2.add(BatchNormalization())
model_2.add(Activation('relu'))
model_2.add(Dropout(0.15))

model_2.add(MaxPooling2D(pool_size=(2, 2))) # 3,3,128   RF:31

model_2.add(SeparableConv2D(10, kernel_size=(5, 5)))  # 5,5,128  RF:39


model_2.add(Flatten())
model_2.add(Activation('softmax'))



model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_2.summary()

_______
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(zoom_range=0.0, 
                             horizontal_flip=False,rotation_range = 20)

from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

def scheduler(epoch,lr):
  if(epoch % 10 > 5):
    return round(0.003*1/(1 + 0.319*epoch),10)
  else:
    return 0.001 +  round(0.003*1/(1 + 0.319*epoch),10)

model_2.compile(loss = 'categorical_crossentropy',optimizer = Adam(lr = 0.003), metrics = ['accuracy'])

# train the model
start = time.time()
# Train the model
model_info = model_2.fit_generator(datagen.flow(train_features, train_labels, batch_size = 128,shuffle=False),
                                 samples_per_epoch = train_features.shape[0], nb_epoch = 50, 
                                 validation_data = (test_features, test_labels), callbacks = [LearningRateScheduler(scheduler, verbose=1)])
end = time.time()
print ("Model took %0.2f seconds to train"%(end - start))
# plot model history
plot_model_history(model_info)
# compute test accuracy
print ("Accuracy on test data is: %0.2f"%accuracy(test_features, test_labels, model_2))


##  Your 50 epoch logs

Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.004.
390/390 [==============================] - 49s 125ms/step - loss: 1.4720 - acc: 0.4589 - val_loss: 1.6871 - val_acc: 0.4838
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0032744503.
390/390 [==============================] - 38s 96ms/step - loss: 1.1100 - acc: 0.6043 - val_loss: 1.3861 - val_acc: 0.5397
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0028315018.
390/390 [==============================] - 38s 96ms/step - loss: 0.9623 - acc: 0.6609 - val_loss: 0.9532 - val_acc: 0.6651
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0025329586000000003.
390/390 [==============================] - 37s 96ms/step - loss: 0.8705 - acc: 0.6909 - val_loss: 0.9467 - val_acc: 0.6687
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0023181019000000002.
390/390 [==============================] - 38s 97ms/step - loss: 0.8112 - acc: 0.7160 - val_loss: 1.1117 - val_acc: 0.6230
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0021560694.
390/390 [==============================] - 38s 97ms/step - loss: 0.7644 - acc: 0.7316 - val_loss: 0.8339 - val_acc: 0.7101
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
390/390 [==============================] - 38s 97ms/step - loss: 0.6855 - acc: 0.7604 - val_loss: 0.6632 - val_acc: 0.7725
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
390/390 [==============================] - 37s 96ms/step - loss: 0.6612 - acc: 0.7721 - val_loss: 0.6846 - val_acc: 0.7588
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
390/390 [==============================] - 37s 96ms/step - loss: 0.6428 - acc: 0.7759 - val_loss: 0.7297 - val_acc: 0.7444
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
390/390 [==============================] - 37s 96ms/step - loss: 0.6272 - acc: 0.7790 - val_loss: 0.6449 - val_acc: 0.7764
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0017159904999999999.
390/390 [==============================] - 37s 95ms/step - loss: 0.6647 - acc: 0.7692 - val_loss: 0.9173 - val_acc: 0.6933
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.001665336.
390/390 [==============================] - 37s 95ms/step - loss: 0.6417 - acc: 0.7771 - val_loss: 0.7389 - val_acc: 0.7431
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0016213753.
390/390 [==============================] - 37s 96ms/step - loss: 0.6285 - acc: 0.7787 - val_loss: 0.6792 - val_acc: 0.7743
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0015828638000000002.
390/390 [==============================] - 38s 97ms/step - loss: 0.6106 - acc: 0.7857 - val_loss: 0.7357 - val_acc: 0.7444
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0015488474.
390/390 [==============================] - 38s 98ms/step - loss: 0.5990 - acc: 0.7881 - val_loss: 0.7334 - val_acc: 0.7433
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0015185825.
390/390 [==============================] - 38s 96ms/step - loss: 0.5860 - acc: 0.7944 - val_loss: 0.6312 - val_acc: 0.7836
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
390/390 [==============================] - 37s 96ms/step - loss: 0.5344 - acc: 0.8142 - val_loss: 0.5954 - val_acc: 0.7951
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
390/390 [==============================] - 37s 96ms/step - loss: 0.5247 - acc: 0.8155 - val_loss: 0.5767 - val_acc: 0.8009
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
390/390 [==============================] - 37s 95ms/step - loss: 0.5104 - acc: 0.8207 - val_loss: 0.5774 - val_acc: 0.8022
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
390/390 [==============================] - 37s 96ms/step - loss: 0.5016 - acc: 0.8243 - val_loss: 0.5743 - val_acc: 0.8018
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0014065041.
390/390 [==============================] - 38s 97ms/step - loss: 0.5489 - acc: 0.8072 - val_loss: 0.6316 - val_acc: 0.7859
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.001389661.
390/390 [==============================] - 38s 96ms/step - loss: 0.5500 - acc: 0.8082 - val_loss: 0.6010 - val_acc: 0.7908
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0013741581.
390/390 [==============================] - 38s 96ms/step - loss: 0.5401 - acc: 0.8107 - val_loss: 0.6282 - val_acc: 0.7833
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0013598417.
390/390 [==============================] - 37s 95ms/step - loss: 0.5318 - acc: 0.8148 - val_loss: 0.6595 - val_acc: 0.7770
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0013465804000000001.
390/390 [==============================] - 37s 96ms/step - loss: 0.5233 - acc: 0.8171 - val_loss: 0.5680 - val_acc: 0.8018
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0013342618.
390/390 [==============================] - 37s 95ms/step - loss: 0.5193 - acc: 0.8182 - val_loss: 0.6198 - val_acc: 0.7887
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0003227889.
390/390 [==============================] - 37s 96ms/step - loss: 0.4703 - acc: 0.8355 - val_loss: 0.5144 - val_acc: 0.8229
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0003120774.
390/390 [==============================] - 37s 95ms/step - loss: 0.4562 - acc: 0.8397 - val_loss: 0.5141 - val_acc: 0.8228
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.000302054.
390/390 [==============================] - 37s 94ms/step - loss: 0.4526 - acc: 0.8420 - val_loss: 0.5175 - val_acc: 0.8203
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0002926544.
390/390 [==============================] - 37s 95ms/step - loss: 0.4475 - acc: 0.8436 - val_loss: 0.5177 - val_acc: 0.8191
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0012838221.
390/390 [==============================] - 37s 95ms/step - loss: 0.5029 - acc: 0.8240 - val_loss: 0.5631 - val_acc: 0.8063
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0012755074.
390/390 [==============================] - 37s 94ms/step - loss: 0.4949 - acc: 0.8256 - val_loss: 0.5716 - val_acc: 0.8042
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.001267666.
390/390 [==============================] - 37s 94ms/step - loss: 0.4936 - acc: 0.8263 - val_loss: 0.5808 - val_acc: 0.7985
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0012602585.
390/390 [==============================] - 37s 94ms/step - loss: 0.4820 - acc: 0.8304 - val_loss: 0.6543 - val_acc: 0.7784
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.00125325.
390/390 [==============================] - 37s 95ms/step - loss: 0.4776 - acc: 0.8314 - val_loss: 0.5591 - val_acc: 0.8075
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0012466091000000001.
390/390 [==============================] - 37s 95ms/step - loss: 0.4782 - acc: 0.8312 - val_loss: 0.5460 - val_acc: 0.8118
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0002403076.
390/390 [==============================] - 37s 95ms/step - loss: 0.4283 - acc: 0.8493 - val_loss: 0.5224 - val_acc: 0.8212
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0002343201.
390/390 [==============================] - 37s 94ms/step - loss: 0.4130 - acc: 0.8558 - val_loss: 0.4946 - val_acc: 0.8294
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0002286237.
390/390 [==============================] - 37s 95ms/step - loss: 0.4164 - acc: 0.8523 - val_loss: 0.4998 - val_acc: 0.8273
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0002231977.
390/390 [==============================] - 37s 95ms/step - loss: 0.4050 - acc: 0.8589 - val_loss: 0.5220 - val_acc: 0.8215
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0012180233.
390/390 [==============================] - 37s 94ms/step - loss: 0.4644 - acc: 0.8354 - val_loss: 0.5766 - val_acc: 0.8029
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0012130833.
390/390 [==============================] - 37s 94ms/step - loss: 0.4637 - acc: 0.8362 - val_loss: 0.5916 - val_acc: 0.8020
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0012083623.
390/390 [==============================] - 37s 94ms/step - loss: 0.4550 - acc: 0.8388 - val_loss: 0.5671 - val_acc: 0.8073
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0012038459000000001.
390/390 [==============================] - 37s 94ms/step - loss: 0.4534 - acc: 0.8398 - val_loss: 0.5853 - val_acc: 0.8009
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0011995211.
390/390 [==============================] - 36s 93ms/step - loss: 0.4459 - acc: 0.8421 - val_loss: 0.5814 - val_acc: 0.8008
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0011953761.
390/390 [==============================] - 36s 93ms/step - loss: 0.4460 - acc: 0.8419 - val_loss: 0.4977 - val_acc: 0.8261
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0001913998.
390/390 [==============================] - 36s 94ms/step - loss: 0.4047 - acc: 0.8563 - val_loss: 0.5042 - val_acc: 0.8274
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0001875821.
390/390 [==============================] - 37s 94ms/step - loss: 0.3951 - acc: 0.8623 - val_loss: 0.5002 - val_acc: 0.8289
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0001839137.
390/390 [==============================] - 36s 93ms/step - loss: 0.3835 - acc: 0.8656 - val_loss: 0.4952 - val_acc: 0.8302
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.000180386.
390/390 [==============================] - 37s 94ms/step - loss: 0.3839 - acc: 0.8657 - val_loss: 0.4779 - val_acc: 0.8344
Model took 1870.85 seconds to train
