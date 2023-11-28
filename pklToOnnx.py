import pickle 
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tf2onnx
import onnx

model = pickle.load(open("DentAIv3.pkl", "rb"))
frozen_graph = convert_variables_to_constants_v2(model)
onnx_model = tf2onnx.convert.from_tensorflow(model)  
onnx.save(onnx_model, "DentAIv3.onnx")