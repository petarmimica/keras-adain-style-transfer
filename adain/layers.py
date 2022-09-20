

import tensorflow as tf
from adain import USE_TF_KERAS


if USE_TF_KERAS:
    Layer = tf.keras.layers.Layer
else:
    import keras
    Layer = keras.layers.Layer


class VggPreprocess(Layer):

    def __init__(self, **kwargs):
        super(VggPreprocess, self).__init__(**kwargs)

    def call(self, x):
        import numpy as np
        x = tf.reverse(x, axis=[-1])
        x = x - tf.constant(np.array([103.939, 116.779, 123.68], dtype=np.float32))
        return x


class PostPreprocess(Layer):
 
    def __init__(self, **kwargs):
        super(PostPreprocess, self).__init__(**kwargs)
 
    def call(self, x):
        x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0)
        x = x * 255
        return x


class SpatialReflectionPadding(Layer):

    def __init__(self, **kwargs):
        super(SpatialReflectionPadding, self).__init__(**kwargs)

    def call(self, x):
        return tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), mode="REFLECT")


class AdaIN(Layer):
    def __init__(self, alpha=1.0, **kwargs):
        self.alpha = alpha
        super(AdaIN, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        assert input_shape[0] == input_shape[1]
        return input_shape[0]

    def call(self, x):
        assert isinstance(x, list)
        # Todo : args
        content_features, style_features = x[0], x[1]
        style_mean, style_variance = tf.nn.moments(style_features, [1,2], keep_dims=True)
        content_mean, content_variance = tf.nn.moments(content_features, [1,2], keep_dims=True)
        epsilon = 1e-5
        normalized_content_features = tf.nn.batch_normalization(content_features, content_mean,
                                                                content_variance, style_mean, 
                                                                tf.sqrt(style_variance), epsilon)
        normalized_content_features = self.alpha * normalized_content_features + (1 - self.alpha) * content_features
        return normalized_content_features

