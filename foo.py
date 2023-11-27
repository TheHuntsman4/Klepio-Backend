import keras
import pickle

model = keras.models.load_model('DentNNv3.h5')
model_pkl = pickle.dumps(model)

with open('DentAIv3.pkl', 'wb') as f:
    f.write(model_pkl)