!pip install tensorflow
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
"""
There are two ways to build Keras models: sequential and functional.
The sequential API allows you to create models layer-by-layer for most problems. It is limited in that it does not allow you to create models that share layers or have multiple inputs or outputs.
Alternatively, the functional API allows you to create models that have a lot more flexibility as you can easily define models where layers connect to more than just the previous and next layers. In fact, you can connect layers to (literally) any other layer. As a result, creating complex networks such as siamese networks and residual networks become possible.
"""
