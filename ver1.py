#!/usr/bin/env python
# coding: utf-8

# In[17]:


from __future__ import print_function
import sys, os, glob, datetime, shutil, traceback
from keras.callbacks import LambdaCallback, Callback, ModelCheckpoint, CSVLogger
from keras.models import Sequential, load_model
from keras.layers import LSTM, CuDNNLSTM, Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.wrappers.scikit_learn import KerasRegressor
from keras import metrics
from keras.utils.data_utils import get_file
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


# In[18]:


from numpy.random import seed
from tensorflow import set_random_seed

seed(1)
set_random_seed(2)

sys.version


# In[19]:


def load_data(filename):
    dataset = pd.read_csv(filename)
    dataset.loc[:, 'time2'] = pd.to_datetime(dataset.loc[:, 'time2'])
    return dataset

def prepare_data(dataset, train_params):
    n_train, n_val, n_test = train_params['n_train'], train_params['n_val'], train_params['n_test']
    serie_len, serie_y_len, sample_step_size = train_params['serie_len'], train_params['serie_y_len'], train_params['sample_step_size']
#     train = dataset.loc[0:500000]
    train = dataset.sort_values("time")

    params_len = len(train.columns)

    avg_s = "avg" + str(serie_y_len)
    per_s = "per" + str(serie_y_len)
    avg_per_s = "avg_per" + str(serie_y_len)

    train.loc[:, avg_s] = train["price_close"].rolling(serie_y_len).mean()
    train.loc[:, per_s] = train[avg_s].pct_change(serie_y_len) * 100
    train.loc[:, avg_per_s] = train[per_s].rolling(serie_y_len).mean()
    train = train.dropna()
    train = train.reset_index()
    
    input_fields = ('price_open', 'price_high',
           'price_low', 'price_close', 'sdelki_sum', 'sdelki_b05', 'sdelki_b4',
           'sdelki_b20', 'sdelki_b40', 'sdelki_b110', 'sdelki_b999', 'sdelki_s05',
           'sdelki_s4', 'sdelki_s20', 'sdelki_s40', 'sdelki_s110', 'sdelki_s999',
           'sdelki_pb05', 'sdelki_pb4', 'sdelki_pb20', 'sdelki_pb40',
           'sdelki_pb110', 'sdelki_pb999', 'sdelki_ps05', 'sdelki_ps4',
           'sdelki_ps20', 'sdelki_ps40', 'sdelki_ps110', 'sdelki_ps999')
    train_features = train.loc[:, input_fields]

    # Normalization
    sc = MinMaxScaler(feature_range=(0,1))
    train_features = sc.fit_transform(train_features)
    
    series = []
    series_y = []

    for i in range(0, len(train_features) - serie_len - serie_y_len, sample_step_size):
        series.append(train_features[i: i + serie_len])
        series_y.append(train.loc[i + serie_len + serie_y_len, avg_per_s])
    
    x = np.zeros((len(series), serie_len, train_features.shape[1]))
    y = np.zeros((len(series), 1))
    
    for i, serie in enumerate(series):
        x[i] = serie
        y[i] = series_y[i]
    
    print('x.shape: ', x.shape)
    
#     np.random.seed(0)
#     perm = np.random.permutation(n_train + n_val)
    data_x, data_y = x, y

    train_x, train_y = data_x[:n_train], data_y[:n_train]
    val_x, val_y = data_x[n_train:n_train+n_val], data_y[n_train:n_train+n_val]
    test_x, test_y = x[-n_test:], y[-n_test:]
    
    return train_x, train_y, val_x, val_y, test_x, test_y


# In[22]:


# Test loss callback
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.test_loss = []
    
    def on_epoch_end(self, epoch, logs={}):
        # Loss history
        _test_loss = self.model.evaluate(test_x, test_y)
        self.test_loss.append(_test_loss)
        print(" - test_loss: %f" % (_test_loss))
        
        df = pd.DataFrame({'epoch': epoch,
                          'loss': [logs['loss']],
                          'val_loss': [logs['val_loss']],
                          'test_loss': [_test_loss]})
        log_filename = os.path.join(volume_dir, 'log_{}.csv'.format(scheme))
        header = True if not os.path.isfile(log_filename) else False
        df.to_csv(log_filename,
                 index=False,
                 mode='a+',
                 header=header)
        
        return


def define_callbacks(checkpoint_path, checkpoint_names, today_date):
    # Checkpoint
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    filepath = os.path.join(checkpoint_path, checkpoint_names)
    checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                          save_weights_only=False,
                                          monitor='val_loss')
    
    # Metrics
    metrics = Metrics()
    
    class SpotTermination(Callback):
        def on_batch_begin(self, batch, logs={}):
            status_code = requests.get("http://169.254.169.254/latest/meta-data/spot/instance-action").status_code
            if status_code != 404:
                time.sleep(150)
    spot_termination_callback = SpotTermination()

    return [checkpoint_callback, metrics, spot_termination_callback]


# In[23]:


def define_model(lstm1_units=32,
                lstm1_dropout=0.5,
                lstm2_units=64,
                lstm2_dropout=0,
		input_shape=()):
    model = Sequential()
    return_sequences = True if lstm2_units > 0 else False
    model.add(CuDNNLSTM(lstm1_units, input_shape=input_shape, return_sequences=return_sequences))
    model.add(Dropout(lstm1_dropout))
    
    if lstm2_units > 0:
        model.add(CuDNNLSTM(lstm2_units))
        model.add(Dropout(lstm2_dropout))
        
    model.add(Dense(1))
    
    return model

def load_checkpoint_model(checkpoint_path, checkpoint_names):
    list_checkpoint_files = glob.glob(os.path.join(checkpoint_path, '*'))
    checkpoint_epoch = max([int(file.split('.')[1]) for file in list_checkpoint_files])
    checkpoint_epoch_path = os.path.join(checkpoint_path, 
                                         checkpoint_names.format(epoch=checkpoint_epoch))
    model = load_model(checkpoint_epoch_path)
    
    return model, checkpoint_epoch


# In[20]:
def main(volume_dir=''):
	# Define parameters
	epochs = 100
	batch_size = 512

	filename = 'convert_sber1.csv'
	train_params = {
	    'serie_len': 50, 
	    'serie_y_len': 50, 
	    'sample_step_size': 10,
	    'n_val': 5000,
	    'n_test': 5000,
	    'n_train': 25000
	}

	scheme = 'LSTM32_D05_64_0__25K_I29_b'

	# Prepare data
	checkpoint_path = os.path.join(volume_dir, 'checkpoints/' + scheme)
	checkpoint_names = 'myvols_model.{epoch:03d}.h5'
	dataset_path = os.path.join(volume_dir, 'datasets')
	today_date = datetime.datetime.today().strftime('%Y-%m-%d')
	dataset = load_data(os.path.join(dataset_path, filename))
	print('dataset.shape', dataset.shape)
	train_x, train_y, val_x, val_y, test_x, test_y = prepare_data(dataset, train_params)
	print('train shape', train_x.shape, train_y.shape)
	print('val shape', val_x.shape, val_y.shape)
	print('test shape', test_x.shape, test_y.shape)
	
	# Define callbacks
	callbacks=define_callbacks(checkpoint_path, checkpoint_names, today_date)

	# Load model
	if os.path.isdir(checkpoint_path) and any(glob.glob(os.path.join(checkpoint_path, '*'))):
	    model, initial_epoch = load_checkpoint_model(checkpoint_path, checkpoint_names)
	else:
	    model = define_model(input_shape=(train_params['serie_len'], train_x.shape[2]))
	    initial_epoch = 0
	
	opt = Adam(lr=0.001)
	model.compile(loss='mse', optimizer=opt)
	
	# Train model
	history = model.fit(train_x, train_y, epochs=epochs, validation_data=(val_x, val_y), 
		             initial_epoch=initial_epoch, 
				batch_size=batch_size,
				callbacks=callbacks)

	# Score trained model.
	test_scores = model.evaluate(test_x, test_y, verbose=1)
	val_scores = model.evaluate(test_x, test_y, verbose=1)
	print('Test loss:', test_scores)
	print('Val loss:', val_scores)

	# Backup terminal output once training is complete
	shutil.copy2('/var/log/cloud-init-output.log', os.path.join(volume_dir,
                                                                'cloud-init-output-{}.log'.format(today_date)))


if __name__ == "__main__":
    volume_dir = '/dltraining'
    try:
        main(volume_dir)
        
    except Exception:
        print(traceback.print_exc())
        today = datetime.datetime.today()
        filename = os.path.join(volume_dir,
                                'errorlog_{}.log'.format(today.strftime('%Y-%m-%d')))
        with open(filename, 'a') as f:
            f.write(today.strftime('%Y-%m-%d %H:%M:%S') + ' > ')
            traceback.print_exc(file=f)
            f.write('---------\n\n')


