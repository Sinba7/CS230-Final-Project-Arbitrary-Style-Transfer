# Matching layers we try to implement: Adaptive Instance Normalization, AdaClip, Histogram Matching and CORAL.

import tensorflow as tf


def AdaIN(content, style, epsilon=1e-5):
    # Adaptive Instance Normalization, modified from https://github.com/eridgd/AdaIN-TF/blob/master/coral.py
    meanC, varC = tf.nn.moments(content, [1, 2], keep_dims=True)
    meanS, varS = tf.nn.moments(style,   [1, 2], keep_dims=True)

    sigmaC = tf.sqrt(tf.add(varC, epsilon))
    sigmaS = tf.sqrt(tf.add(varS, epsilon))
    
    return (content - meanC) * sigmaS / sigmaC + meanS


def AdaClip(content, style, epsilon=1e-5):
    # shifting and rescaling content feature so that it has the same min/max
    # as style feature. In this way relu features will still has non-negative
    # values.
    content_min = tf.reduce_min(content, [1, 2], keep_dims=True)
    content_max = tf.reduce_max(content, [1, 2], keep_dims=True)
    style_min = tf.reduce_min(style, [1, 2], keep_dims=True)
    style_max = tf.reduce_max(style, [1, 2], keep_dims=True)
    scale = (style_max - style_min) / (content_max - content_min + epsilon)
    shift = style_min - content_min * scale
    return content * scale + shift


def HistoMatching(content, style, bins=100):
    # doing
    raise NotImplementedError
    # get info of style histogram
    style_range = tf.constant([tf.reduce_min(style), tf.reduce_max(style)], dtype = tf.float32)
    style_histogram = tf.histogram_fixed_width(style, style_range, bins=bins)
    style_cdf = tf.cumsum(style_histogram)
    style_shape = tf.shape(style)

    histogram = tf.histogram_fixed_width(tf.to_float(image), values_range, 256)
    cdf = tf.cumsum(histogram)
    cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]

    img_shape = tf.shape(image)
    pix_cnt = img_shape[-3] * img_shape[-2]
    px_map = tf.round(tf.to_float(cdf - cdf_min) * 255. / tf.to_float(pix_cnt - 1))
    px_map = tf.cast(px_map, tf.uint8)

    eq_hist = tf.expand_dims(tf.gather_nd(px_map, tf.cast(image, tf.int32)), 2)
    return eq_hist


def CORAL(content, style, epsilon=1e-5):
    # Correlation Alignment, modified from https://github.com/eridgd/AdaIN-TF/blob/master/coral.py
    def tf_matrix_sqrt(x):
        # compute square root of matrix x so that tf.matmul(res, res) = x
        s, u, v = tf.svd(x)
        sqrt_x = tf.matmul(tf.matmul(u, tf.sqrt(tf.matrix_diag(s))), v)
        return sqrt_x

    content_shape = tf.shape(content)
    style_shape = tf.shape(style)

    content_flatten = tf.reshape(tf.transpose(content, [0, 3, 1, 2]), 
        [content_shape[0], content_shape[3], content_shape[1]*content_shape[2]])  # (N * C * HW)
    style_flatten = tf.reshape(tf.transpose(style, [0, 3, 1, 2]), 
        [style_shape[0], style_shape[3], style_shape[1]*style_shape[2]])  # N * C * HW

    content_mean, content_var = tf.nn.moments(content_flatten, axes=[-1], keep_dims=True)
    style_mean, style_var = tf.nn.moments(style_flatten, axes=[-1], keep_dims=True)

    content_std = tf.sqrt(tf.add(content_var, epsilon))
    style_std = tf.sqrt(tf.add(style_var, epsilon))

    content_flatten_norm = (content_flatten - content_mean) / content_std  # N * C * HW
    style_flatten_norm = (style_flatten - style_mean) / style_std

    content_flatten_cov_eye = tf.matmul(content_flatten_norm, 
        tf.transpose(content_flatten_norm, [0, 2, 1])) + tf.tile(tf.expand_dims(tf.eye(content_shape[3]), 0), 
        [content_shape[0], 1, 1])  # N * C * C
    style_flatten_cov_eye = tf.matmul(style_flatten_norm, 
        tf.transpose(style_flatten_norm, [0, 2, 1])) + tf.tile(tf.expand_dims(tf.eye(style_shape[3]), 0), 
        [style_shape[0], 1, 1])  # N * C * C

    content_flatten_norm_transfer = tf.matmul( tf_matrix_sqrt(style_flatten_cov_eye), tf.matmul(tf.matrix_inverse(tf_matrix_sqrt(content_flatten_cov_eye)), content_flatten_norm))
    content_flatten_transfer = content_flatten_norm_transfer * style_std + style_mean

    content_transfer = tf.transpose(tf.reshape(content_flatten_transfer, 
        [content_shape[0], content_shape[3], content_shape[1], content_shape[2]]), [0, 2, 3, 1])  # N*H*W*C
    return content_transfer
