#!/usr/bin/env python
from model import *
from data import *
from tqdm.auto import tqdm



def train():
    
    # print('Compiling model...     ')
    model, model_checkpoint, tensorboard = get_model('training')
    model.load_weights(r'/home/dl/Codes/3DSuper/Outputs/SuperResolution/weights/your_prediction/weights.h5', by_name=True)
    
    # print('Load training data...  ')
    training_data_input, training_data_pred_ground_truth = load_train_data()
    
    # print('Training model...      ')
    model.fit(training_data_input,
              training_data_pred_ground_truth,
              batch_size=20,
              validation_split=0.1,
              epochs=50,
              verbose=1,
              shuffle=True,
              callbacks=[model_checkpoint, tensorboard]
             )

    model.save(weight_name)


def MC_drop(iterate_count=50):

    test_data_input, _ = load_test_data()
    # load model
    model, _, _ = get_model('testing')
    model.load_weights(weight_name)

    this_test = np.empty([iterate_count, image_rows_low, image_cols, channel_num], dtype=np.float32)
    test_data_prediction = np.empty([test_data_input.shape[0], image_rows_high, image_cols, 2], dtype=np.float32)

    for i in tqdm(range(test_data_prediction.shape[0])):

#         print('Processing {} th of {} images ... '.format(i, test_data_prediction.shape[0]))
        
        for j in range(iterate_count):
            this_test[j] = test_data_input[i]

        this_prediction = model.predict(this_test, verbose=1)

        this_prediction_mean = np.mean(this_prediction, axis=0)
        this_prediction_var = np.std(this_prediction, axis=0)
        test_data_prediction[i,:,:,0:1] = this_prediction_mean
        test_data_prediction[i,:,:,1:2] = this_prediction_var

#     np.save('/home/dl/Codes/3DSuper/Outputs/SuperResolution/Result/' + 'prediction.npy', test_data_prediction)


if __name__ == '__main__':

    # -> train network
    train()

    # -> Monte-Carlo Dropout Test
#     MC_drop()
    
