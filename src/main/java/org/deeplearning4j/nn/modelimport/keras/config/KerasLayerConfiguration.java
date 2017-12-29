/*
 * Copyright (c) 2017 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.deeplearning4j.nn.modelimport.keras.config;


/**
 * All relevant property fields of keras layers.
 * <p>
 * Empty String fields mean Keras 1 and 2 implementations differ,
 * supplied fields stand for shared properties.
 *
 * @author Max Pumperla
 */
public class KerasLayerConfiguration {
  
  private final String LAYER_FIELD_KERAS_VERSION = "keras_version";
  private final String LAYER_FIELD_CLASS_NAME = "class_name";
  private final String LAYER_FIELD_LAYER = "layer";
  
  /* Basic layer names */
  // Missing Layers: Permute, RepeatVector, Lambda, ActivityRegularization, Masking
  // Conv3D, Cropping1D-3D, UpSampling3D,
  // ZeroPadding3D, LocallyConnected1D-2D
  // Missing layers from keras 1: Highway, MaxoutDense
  private final String LAYER_CLASS_NAME_ACTIVATION = "Activation";
  private final String LAYER_CLASS_NAME_INPUT = "InputLayer";
  private final String LAYER_CLASS_NAME_DROPOUT = "Dropout";
  private final String LAYER_CLASS_NAME_ALPHA_DROPOUT = "AlphaDropout";
  private final String LAYER_CLASS_NAME_GAUSSIAN_DROPOUT = "GaussianDropout";
  private final String LAYER_CLASS_NAME_GAUSSIAN_NOISE = "GaussianNoise";
  private final String LAYER_CLASS_NAME_DENSE = "Dense";
  
  private final String LAYER_CLASS_NAME_LSTM = "LSTM";
  private final String LAYER_CLASS_NAME_SIMPLE_RNN = "SimpleRNN";
  
  private final String LAYER_CLASS_NAME_BIDIRECTIONAL = "Bidirectional";
  private final String LAYER_CLASS_NAME_TIME_DISTRIBUTED = "TimeDistributed";
  
  
  private final String LAYER_CLASS_NAME_MAX_POOLING_1D = "MaxPooling1D";
  private final String LAYER_CLASS_NAME_MAX_POOLING_2D = "MaxPooling2D";
  private final String LAYER_CLASS_NAME_AVERAGE_POOLING_1D = "AveragePooling1D";
  private final String LAYER_CLASS_NAME_AVERAGE_POOLING_2D = "AveragePooling2D";
  private final String LAYER_CLASS_NAME_ZERO_PADDING_1D = "ZeroPadding1D";
  private final String LAYER_CLASS_NAME_ZERO_PADDING_2D = "ZeroPadding2D";
  
  private final String LAYER_CLASS_NAME_FLATTEN = "Flatten";
  private final String LAYER_CLASS_NAME_RESHAPE = "Reshape";
  private final String LAYER_CLASS_NAME_MERGE = "Merge";
  
  private final String LAYER_CLASS_NAME_BATCHNORMALIZATION = "BatchNormalization";
  private final String LAYER_CLASS_NAME_EMBEDDING = "Embedding";
  private final String LAYER_CLASS_NAME_GLOBAL_MAX_POOLING_1D = "GlobalMaxPooling1D";
  private final String LAYER_CLASS_NAME_GLOBAL_MAX_POOLING_2D = "GlobalMaxPooling2D";
  private final String LAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_1D = "GlobalAveragePooling1D";
  private final String LAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_2D = "GlobalAveragePooling2D";
  private final String LAYER_CLASS_NAME_TIME_DISTRIBUTED_DENSE = "TimeDistributedDense"; // Keras 1 only
  private final String LAYER_CLASS_NAME_ATROUS_CONVOLUTION_1D = "AtrousConvolution1D"; // Keras 1 only
  private final String LAYER_CLASS_NAME_ATROUS_CONVOLUTION_2D = "AtrousConvolution2D"; // Keras 1 only
  private final String LAYER_CLASS_NAME_CONVOLUTION_1D = ""; // 1: Convolution1D, 2: Conv1D
  private final String LAYER_CLASS_NAME_CONVOLUTION_2D = ""; // 1: Convolution2D, 2: Conv2D
  private final String LAYER_CLASS_NAME_LEAKY_RELU = "LeakyReLU";
  private final String LAYER_CLASS_NAME_UPSAMPLING_1D = "UpSampling1D";
  private final String LAYER_CLASS_NAME_UPSAMPLING_2D = "UpSampling2D";
  private final String LAYER_CLASS_NAME_SEPARABLE_CONVOLUTION_2D = ""; // 1: SeparableConvolution2D, 2: SeparableConv2D
  private final String LAYER_CLASS_NAME_DECONVOLUTION_2D = ""; // 1: Deconvolution2D, 2: Conv2DTranspose
  
  /* Partially shared layer configurations. */
  private final String LAYER_FIELD_INPUT_SHAPE = "input_shape";
  private final String LAYER_FIELD_CONFIG = "config";
  private final String LAYER_FIELD_NAME = "name";
  private final String LAYER_FIELD_BATCH_INPUT_SHAPE = "batch_input_shape";
  private final String LAYER_FIELD_INBOUND_NODES = "inbound_nodes";
  private final String LAYER_FIELD_OUTBOUND_NODES = "outbound_nodes";
  private final String LAYER_FIELD_DROPOUT = "dropout";
  private final String LAYER_FIELD_ACTIVITY_REGULARIZER = "activity_regularizer";
  private final String LAYER_FIELD_EMBEDDING_OUTPUT_DIM = "output_dim";
  private final String LAYER_FIELD_OUTPUT_DIM = ""; // 1: output_dim, 2: units
  private final String LAYER_FIELD_DROPOUT_RATE = ""; // 1: p, 2: rate
  private final String LAYER_FIELD_USE_BIAS = ""; // 1: bias, 2: use_bias
  private final String KERAS_PARAM_NAME_W = ""; // 1: W, 2: kernel
  private final String KERAS_PARAM_NAME_B = ""; // 1: b, 2: bias
  private final String KERAS_PARAM_NAME_RW = ""; // 1: U, 2: recurrent_kernel
  
  
  /* Keras dimension ordering for, e.g., convolutional layersOrdered. */
  private final String LAYER_FIELD_BACKEND = "backend"; // not available in keras 1, caught in code
  private final String LAYER_FIELD_DIM_ORDERING = ""; // 1: dim_ordering, 2: data_format
  private final String DIM_ORDERING_THEANO = ""; // 1: th, 2: channels_first
  private final String DIM_ORDERING_TENSORFLOW = ""; // 1: tf, 2: channels_last
  
  /* Recurrent layers */
  private final String LAYER_FIELD_DROPOUT_W = ""; // 1: dropout_W, 2: dropout
  private final String LAYER_FIELD_DROPOUT_U = ""; // 2: dropout_U, 2: recurrent_dropout
  private final String LAYER_FIELD_INNER_INIT = ""; // 1: inner_init, 2: recurrent_initializer
  private final String LAYER_FIELD_RECURRENT_CONSTRAINT = "recurrent_constraint"; // keras 2 only
  private final String LAYER_FIELD_RECURRENT_DROPOUT = ""; // 1: dropout_U, 2: recurrent_dropout
  private final String LAYER_FIELD_INNER_ACTIVATION = ""; // 1: inner_activation, 2: recurrent_activation
  private final String LAYER_FIELD_FORGET_BIAS_INIT = "forget_bias_init"; // keras 1 only: string
  private final String LAYER_FIELD_UNIT_FORGET_BIAS = "unit_forget_bias"; // keras 1 only: bool
  private final String LAYER_FIELD_RETURN_SEQUENCES = "return_sequences";
  private final String LAYER_FIELD_UNROLL = "unroll";
  
  /* Embedding layer properties */
  private final String LAYER_FIELD_INPUT_DIM = "input_dim";
  private final String LAYER_FIELD_EMBEDDING_INIT = ""; // 1: "init", 2: "embeddings_initializer"
  private final String LAYER_FIELD_EMBEDDING_WEIGHTS = ""; // 1: "W", 2: "embeddings"
  private final String LAYER_FIELD_EMBEDDINGS_REGULARIZER = ""; // 1: W_regularizer, 2: embeddings_regularizer
  private final String LAYER_FIELD_EMBEDDINGS_CONSTRAINT = ""; // 1: W_constraint, 2: embeddings_constraint
  private final String LAYER_FIELD_MASK_ZERO = "mask_zero";
  private final String LAYER_FIELD_INPUT_LENGTH = "input_length";
  
  /* Keras separable convolution types */
  private final String LAYER_PARAM_NAME_DEPTH_WISE_KERNEL = "depthwise_kernel";
  private final String LAYER_PARAM_NAME_POINT_WISE_KERNEL = "pointwise_kernel";
  
  private final String LAYER_FIELD_DEPTH_WISE_INIT = "depthwise_initializer";
  private final String LAYER_FIELD_POINT_WISE_INIT = "pointwise_initializer";
  
  private final String LAYER_FIELD_DEPTH_WISE_REGULARIZER = "depthwise_regularizer";
  private final String LAYER_FIELD_POINT_WISE_REGULARIZER = "pointwise_regularizer";
  
  private final String LAYER_FIELD_DEPTH_WISE_CONSTRAINT = "depthwise_constraint";
  private final String LAYER_FIELD_POINT_WISE_CONSTRAINT = "pointwise_constraint";
  
  /* Normalisation layers */
  // Missing: keras 2 moving_mean_initializer, moving_variance_initializer
  private final String LAYER_FIELD_BATCHNORMALIZATION_MODE = "mode"; // keras 1 only
  private final String LAYER_FIELD_BATCHNORMALIZATION_BETA_INIT = ""; // 1: beta_init, 2: beta_initializer
  private final String LAYER_FIELD_BATCHNORMALIZATION_GAMMA_INIT = ""; // 1: gamma_init, 2: gamma_initializer
  private final String LAYER_FIELD_BATCHNORMALIZATION_BETA_CONSTRAINT = "beta_constraint"; // keras 2 only
  private final String LAYER_FIELD_BATCHNORMALIZATION_GAMMA_CONSTRAINT = "gamma_constraint"; // keras 2 only
  private final String LAYER_FIELD_BATCHNORMALIZATION_MOVING_MEAN = ""; // 1: running_mean, 2: moving_mean
  private final String LAYER_FIELD_BATCHNORMALIZATION_MOVING_VARIANCE = ""; // 1: running_std, 2: moving_variance
  
  /* Advanced activations */
  // Missing: LeakyReLU, PReLU, ThresholdedReLU, ParametricSoftplus, SReLu
  private final String LAYER_FIELD_PRELU_INIT = ""; // 1: init, 2: alpha_initializer
  
  /* Convolutional layer properties */
  private final String LAYER_FIELD_NB_FILTER = ""; // 1: nb_filter, 2: filters
  private final String LAYER_FIELD_NB_ROW = "nb_row"; // keras 1 only
  private final String LAYER_FIELD_NB_COL = "nb_col"; // keras 1 only
  private final String LAYER_FIELD_KERNEL_SIZE = "kernel_size"; // keras 2 only
  private final String LAYER_FIELD_POOL_SIZE = "pool_size";
  private final String LAYER_FIELD_CONVOLUTION_STRIDES = ""; // 1: subsample, 2: strides
  private final String LAYER_FIELD_FILTER_LENGTH = ""; // 1: filter_length, 2: kernel_size
  private final String LAYER_FIELD_SUBSAMPLE_LENGTH = ""; // 1: subsample_length, 2: strides
  private final String LAYER_FIELD_DILATION_RATE = ""; // 1: atrous_rate, 2: dilation_rate
  private final String LAYER_FIELD_ZERO_PADDING = "padding";
  
  /* Pooling / Upsampling layer properties */
  private final String LAYER_FIELD_POOL_STRIDES = "strides";
  private final String LAYER_FIELD_POOL_1D_SIZE = ""; // 1: pool_length, 2: pool_size
  private final String LAYER_FIELD_POOL_1D_STRIDES = ""; // 1: stride, 2: strides
  private final String LAYER_FIELD_UPSAMPLING_1D_SIZE = ""; // 1: length, 2: size
  private final String LAYER_FIELD_UPSAMPLING_2D_SIZE = "size";
  
  
  /* Keras convolution border modes. */
  private final String LAYER_FIELD_BORDER_MODE = ""; // 1: border_mode, 2: padding
  private final String LAYER_BORDER_MODE_SAME = "same";
  private final String LAYER_BORDER_MODE_VALID = "valid";
  private final String LAYER_BORDER_MODE_FULL = "full";
  
  /* Noise layers */
  private final String LAYER_FIELD_RATE = "rate";
  private final String LAYER_FIELD_GAUSSIAN_VARIANCE = ""; // 1: sigma, 2: stddev

    /* Layer wrappers */
  // Missing: TimeDistributed
  
  
  /* Keras weight regularizers. */
  private final String LAYER_FIELD_W_REGULARIZER = ""; // 1: W_regularizer, 2: kernel_regularizer
  private final String LAYER_FIELD_B_REGULARIZER = ""; // 1: b_regularizer, 2: bias_regularizer
  private final String REGULARIZATION_TYPE_L1 = "l1";
  private final String REGULARIZATION_TYPE_L2 = "l2";
  
  /* Keras constraints */
  private final String LAYER_FIELD_MINMAX_NORM_CONSTRAINT = "MinMaxNorm";
  private final String LAYER_FIELD_MINMAX_NORM_CONSTRAINT_ALIAS = "min_max_norm";
  private final String LAYER_FIELD_MAX_NORM_CONSTRAINT = "MaxNorm";
  private final String LAYER_FIELD_MAX_NORM_CONSTRAINT_ALIAS = "max_norm";
  private final String LAYER_FIELD_MAX_NORM_CONSTRAINT_ALIAS_2 = "maxnorm";
  private final String LAYER_FIELD_NON_NEG_CONSTRAINT = "NonNeg";
  private final String LAYER_FIELD_NON_NEG_CONSTRAINT_ALIAS = "nonneg";
  private final String LAYER_FIELD_NON_NEG_CONSTRAINT_ALIAS_2 = "non_neg";
  private final String LAYER_FIELD_UNIT_NORM_CONSTRAINT = "UnitNorm";
  private final String LAYER_FIELD_UNIT_NORM_CONSTRAINT_ALIAS = "unitnorm";
  private final String LAYER_FIELD_UNIT_NORM_CONSTRAINT_ALIAS_2 = "unit_norm";
  private final String LAYER_FIELD_CONSTRAINT_NAME = ""; // 1: name, 2: class_name
  private final String LAYER_FIELD_W_CONSTRAINT = ""; // 1: W_constraint, 2: kernel_constraint
  private final String LAYER_FIELD_B_CONSTRAINT = ""; // 1: b_constraint, 2: bias_constraint
  private final String LAYER_FIELD_MAX_CONSTRAINT = ""; // 1: m, 2: max_value
  private final String LAYER_FIELD_MINMAX_MIN_CONSTRAINT = ""; // 1: low, 2: min_value
  private final String LAYER_FIELD_MINMAX_MAX_CONSTRAINT = ""; // 1: high, 2: max_value
  private final String LAYER_FIELD_CONSTRAINT_DIM = "axis";
  private final String LAYER_FIELD_CONSTRAINT_RATE = "rate";
  
  
  /* Keras weight initializers. */
  private final String LAYER_FIELD_INIT = ""; // 1: init, 2: kernel_initializer
  private final String LAYER_FIELD_BIAS_INIT = "bias_initializer"; // keras 2 only
  private final String LAYER_FIELD_INIT_MEAN = "mean";
  private final String LAYER_FIELD_INIT_STDDEV = "stddev";
  private final String LAYER_FIELD_INIT_SCALE = "scale";
  private final String LAYER_FIELD_INIT_MINVAL = "minval";
  private final String LAYER_FIELD_INIT_MAXVAL = "maxval";
  private final String LAYER_FIELD_INIT_VALUE = "value";
  private final String LAYER_FIELD_INIT_GAIN = "gain";
  private final String LAYER_FIELD_INIT_MODE = "mode";
  private final String LAYER_FIELD_INIT_DISTRIBUTION = "distribution";
  
  private final String INIT_UNIFORM = "uniform";
  private final String INIT_RANDOM_UNIFORM = "random_uniform";
  private final String INIT_RANDOM_UNIFORM_ALIAS = "RandomUniform";
  private final String INIT_ZERO = "zero";
  private final String INIT_ZEROS = "zeros";
  private final String INIT_ZEROS_ALIAS = "Zeros";
  private final String INIT_ONE = "one";
  private final String INIT_ONES = "ones";
  private final String INIT_ONES_ALIAS = "Ones";
  private final String INIT_CONSTANT = "constant";
  private final String INIT_CONSTANT_ALIAS = "Constant";
  private final String INIT_TRUNCATED_NORMAL = "truncated_normal";
  private final String INIT_TRUNCATED_NORMAL_ALIAS = "TruncatedNormal";
  private final String INIT_GLOROT_NORMAL = "glorot_normal";
  private final String INIT_GLOROT_UNIFORM = "glorot_uniform";
  private final String INIT_HE_NORMAL = "he_normal";
  private final String INIT_HE_UNIFORM = "he_uniform";
  private final String INIT_LECUN_UNIFORM = "lecun_uniform";
  private final String INIT_LECUN_NORMAL = "lecun_normal";
  private final String INIT_NORMAL = "normal";
  private final String INIT_RANDOM_NORMAL = "random_normal";
  private final String INIT_RANDOM_NORMAL_ALIAS = "RandomNormal";
  private final String INIT_ORTHOGONAL = "orthogonal";
  private final String INIT_ORTHOGONAL_ALIAS = "Orthogonal";
  private final String INIT_IDENTITY = "identity";
  private final String INIT_IDENTITY_ALIAS = "Identity";
  private final String INIT_VARIANCE_SCALING = "VarianceScaling"; // keras 2 only
  
  
  /* Keras and DL4J activation types. */
  private final String LAYER_FIELD_ACTIVATION = "activation";
  
  private final String KERAS_ACTIVATION_SOFTMAX = "softmax";
  private final String KERAS_ACTIVATION_SOFTPLUS = "softplus";
  private final String KERAS_ACTIVATION_SOFTSIGN = "softsign";
  private final String KERAS_ACTIVATION_RELU = "relu";
  private final String KERAS_ACTIVATION_TANH = "tanh";
  private final String KERAS_ACTIVATION_SIGMOID = "sigmoid";
  private final String KERAS_ACTIVATION_HARD_SIGMOID = "hard_sigmoid";
  private final String KERAS_ACTIVATION_LINEAR = "linear";
  private final String KERAS_ACTIVATION_ELU = "elu"; // keras 2 only
  private final String KERAS_ACTIVATION_SELU = "selu"; // keras 2 only
  
  /* Keras loss functions. */
  private final String KERAS_LOSS_MEAN_SQUARED_ERROR = "mean_squared_error";
  private final String KERAS_LOSS_MSE = "mse";
  private final String KERAS_LOSS_MEAN_ABSOLUTE_ERROR = "mean_absolute_error";
  private final String KERAS_LOSS_MAE = "mae";
  private final String KERAS_LOSS_MEAN_ABSOLUTE_PERCENTAGE_ERROR = "mean_absolute_percentage_error";
  private final String KERAS_LOSS_MAPE = "mape";
  private final String KERAS_LOSS_MEAN_SQUARED_LOGARITHMIC_ERROR = "mean_squared_logarithmic_error";
  private final String KERAS_LOSS_MSLE = "msle";
  private final String KERAS_LOSS_SQUARED_HINGE = "squared_hinge";
  private final String KERAS_LOSS_HINGE = "hinge";
  private final String KERAS_LOSS_CATEGORICAL_HINGE = "categorical_hinge"; // keras 2 only
  private final String KERAS_LOSS_BINARY_CROSSENTROPY = "binary_crossentropy";
  private final String KERAS_LOSS_CATEGORICAL_CROSSENTROPY = "categorical_crossentropy";
  private final String KERAS_LOSS_SPARSE_CATEGORICAL_CROSSENTROPY = "sparse_categorical_crossentropy";
  private final String KERAS_LOSS_KULLBACK_LEIBLER_DIVERGENCE = "kullback_leibler_divergence";
  private final String KERAS_LOSS_KLD = "kld";
  private final String KERAS_LOSS_POISSON = "poisson";
  private final String KERAS_LOSS_COSINE_PROXIMITY = "cosine_proximity";
  private final String KERAS_LOSS_LOG_COSH = "logcosh"; // keras 2 only
  
  public String getLAYER_FIELD_BATCHNORMALIZATION_MOVING_MEAN() {
    return LAYER_FIELD_BATCHNORMALIZATION_MOVING_MEAN;
  }
  
  public String getLAYER_FIELD_BATCHNORMALIZATION_MOVING_VARIANCE() {
    return LAYER_FIELD_BATCHNORMALIZATION_MOVING_VARIANCE;
  }
  
  public String getLAYER_FIELD_EMBEDDING_WEIGHTS() {
    return LAYER_FIELD_EMBEDDING_WEIGHTS;
  }
  
  public String getLAYER_FIELD_INPUT_DIM() {
    return LAYER_FIELD_INPUT_DIM;
  }
  
  public String getKERAS_PARAM_NAME_W() {
    return KERAS_PARAM_NAME_W;
  }
  
  public String getKERAS_PARAM_NAME_B() {
    return KERAS_PARAM_NAME_B;
  }
  
  public String getLAYER_FIELD_RETURN_SEQUENCES() {
    return LAYER_FIELD_RETURN_SEQUENCES;
  }
  
  public String getLAYER_FIELD_INNER_INIT() {
    return LAYER_FIELD_INNER_INIT;
  }
  
  public String getLAYER_FIELD_B_CONSTRAINT() {
    return LAYER_FIELD_B_CONSTRAINT;
  }
  
  public String getLAYER_FIELD_W_CONSTRAINT() {
    return LAYER_FIELD_W_CONSTRAINT;
  }
  
  public String getLAYER_FIELD_RECURRENT_CONSTRAINT() {
    return LAYER_FIELD_RECURRENT_CONSTRAINT;
  }
  
  public String getKERAS_PARAM_NAME_RW() {
    return KERAS_PARAM_NAME_RW;
  }
  
  public String getLAYER_FIELD_UNIT_FORGET_BIAS() {
    return LAYER_FIELD_UNIT_FORGET_BIAS;
  }
  
  public String getLAYER_FIELD_FORGET_BIAS_INIT() {
    return LAYER_FIELD_FORGET_BIAS_INIT;
  }
  
  public String getLAYER_FIELD_INNER_ACTIVATION() {
    return LAYER_FIELD_INNER_ACTIVATION;
  }
  
  public String getLAYER_FIELD_INIT() {
    return LAYER_FIELD_INIT;
  }
  
  public String getLAYER_FIELD_DEPTH_WISE_CONSTRAINT() {
    return LAYER_FIELD_DEPTH_WISE_CONSTRAINT;
  }
  
  public String getLAYER_FIELD_POINT_WISE_CONSTRAINT() {
    return LAYER_FIELD_POINT_WISE_CONSTRAINT;
  }
  
  public String getLAYER_FIELD_DEPTH_WISE_REGULARIZER() {
    return LAYER_FIELD_DEPTH_WISE_REGULARIZER;
  }
  
  public String getREGULARIZATION_TYPE_L1() {
    return REGULARIZATION_TYPE_L1;
  }
  
  public String getREGULARIZATION_TYPE_L2() {
    return REGULARIZATION_TYPE_L2;
  }
  
  public String getLAYER_PARAM_NAME_DEPTH_WISE_KERNEL() {
    return LAYER_PARAM_NAME_DEPTH_WISE_KERNEL;
  }
  
  public String getLAYER_PARAM_NAME_POINT_WISE_KERNEL() {
    return LAYER_PARAM_NAME_POINT_WISE_KERNEL;
  }
  
  public String getLAYER_FIELD_DEPTH_WISE_INIT() {
    return this.LAYER_FIELD_DEPTH_WISE_INIT;
  }
  
  public String getLAYER_FIELD_MINMAX_NORM_CONSTRAINT() {
    return LAYER_FIELD_MINMAX_NORM_CONSTRAINT;
  }
  
  public String getLAYER_FIELD_MINMAX_NORM_CONSTRAINT_ALIAS() {
    return LAYER_FIELD_MINMAX_NORM_CONSTRAINT_ALIAS;
  }
  
  public String getLAYER_FIELD_MINMAX_MIN_CONSTRAINT() {
    return LAYER_FIELD_MINMAX_MIN_CONSTRAINT;
  }
  
  public String getLAYER_FIELD_MINMAX_MAX_CONSTRAINT() {
    return LAYER_FIELD_MINMAX_MAX_CONSTRAINT;
  }
  
  public String getLAYER_FIELD_CONSTRAINT_RATE() {
    return LAYER_FIELD_CONSTRAINT_RATE;
  }
  
  public String getLAYER_FIELD_CONSTRAINT_DIM() {
    return LAYER_FIELD_CONSTRAINT_DIM;
  }
  
  public String getLAYER_FIELD_MAX_NORM_CONSTRAINT() {
    return LAYER_FIELD_MAX_NORM_CONSTRAINT;
  }
  
  public String getLAYER_FIELD_MAX_NORM_CONSTRAINT_ALIAS() {
    return LAYER_FIELD_MAX_NORM_CONSTRAINT_ALIAS;
  }
  
  public String getLAYER_FIELD_MAX_NORM_CONSTRAINT_ALIAS_2() {
    return LAYER_FIELD_MAX_NORM_CONSTRAINT_ALIAS_2;
  }
  
  public String getLAYER_FIELD_MAX_CONSTRAINT() {
    return LAYER_FIELD_MAX_CONSTRAINT;
  }
  
  public String getLAYER_FIELD_UNIT_NORM_CONSTRAINT() {
    return LAYER_FIELD_UNIT_NORM_CONSTRAINT;
  }
  
  public String getLAYER_FIELD_UNIT_NORM_CONSTRAINT_ALIAS() {
    return LAYER_FIELD_UNIT_NORM_CONSTRAINT_ALIAS;
  }
  
  public String getLAYER_FIELD_UNIT_NORM_CONSTRAINT_ALIAS_2() {
    return LAYER_FIELD_UNIT_NORM_CONSTRAINT_ALIAS_2;
  }
  
  public String getLAYER_FIELD_NON_NEG_CONSTRAINT() {
    return LAYER_FIELD_NON_NEG_CONSTRAINT;
  }
  
  public String getLAYER_FIELD_NON_NEG_CONSTRAINT_ALIAS() {
    return LAYER_FIELD_NON_NEG_CONSTRAINT_ALIAS;
  }
  
  public String getLAYER_FIELD_NON_NEG_CONSTRAINT_ALIAS_2() {
    return LAYER_FIELD_NON_NEG_CONSTRAINT_ALIAS_2;
  }
  
  public String getLAYER_FIELD_CONSTRAINT_NAME() {
    return LAYER_FIELD_CONSTRAINT_NAME;
  }
  
  public String getLAYER_FIELD_RATE() {
    return LAYER_FIELD_RATE;
  }
  
  public String getLAYER_FIELD_GAUSSIAN_VARIANCE() {
    return LAYER_FIELD_GAUSSIAN_VARIANCE;
  }
  
  public String getINIT_GLOROT_NORMAL() {
    return INIT_GLOROT_NORMAL;
  }
  
  public String getLAYER_FIELD_EMBEDDING_INIT() {
    return LAYER_FIELD_EMBEDDING_INIT;
  }
  
  public String getINIT_GLOROT_UNIFORM() {
    return INIT_GLOROT_UNIFORM;
  }
  
  public String getINIT_LECUN_NORMAL() {
    return INIT_LECUN_NORMAL;
  }
  
  public String getINIT_LECUN_UNIFORM() {
    return INIT_LECUN_UNIFORM;
  }
  
  public String getINIT_HE_NORMAL() {
    return INIT_HE_NORMAL;
  }
  
  public String getINIT_HE_UNIFORM() {
    return INIT_HE_UNIFORM;
  }
  
  public String getINIT_ONE() {
    return INIT_ONE;
  }
  
  public String getINIT_ONES() {
    return INIT_ONES;
  }
  
  public String getINIT_ONES_ALIAS() {
    return INIT_ONES_ALIAS;
  }
  
  public String getINIT_ZERO() {
    return INIT_ZERO;
  }
  
  public String getINIT_ZEROS() {
    return INIT_ZEROS;
  }
  
  public String getINIT_ZEROS_ALIAS() {
    return INIT_ZEROS_ALIAS;
  }
  
  public String getINIT_UNIFORM() {
    return INIT_UNIFORM;
  }
  
  public String getINIT_RANDOM_UNIFORM() {
    return INIT_RANDOM_UNIFORM;
  }
  
  public String getINIT_RANDOM_UNIFORM_ALIAS() {
    return INIT_RANDOM_UNIFORM_ALIAS;
  }
  
  public String getLAYER_FIELD_INIT_MINVAL() {
    return LAYER_FIELD_INIT_MINVAL;
  }
  
  public String getLAYER_FIELD_INIT_MAXVAL() {
    return LAYER_FIELD_INIT_MAXVAL;
  }
  
  public String getLAYER_FIELD_INIT_SCALE() {
    return LAYER_FIELD_INIT_SCALE;
  }
  
  public String getINIT_NORMAL() {
    return INIT_NORMAL;
  }
  
  public String getINIT_RANDOM_NORMAL() {
    return INIT_RANDOM_NORMAL;
  }
  
  public String getINIT_RANDOM_NORMAL_ALIAS() {
    return INIT_RANDOM_NORMAL_ALIAS;
  }
  
  public String getLAYER_FIELD_INIT_MEAN() {
    return LAYER_FIELD_INIT_MEAN;
  }
  
  public String getLAYER_FIELD_INIT_STDDEV() {
    return LAYER_FIELD_INIT_STDDEV;
  }
  
  public String getINIT_CONSTANT() {
    return INIT_CONSTANT;
  }
  
  public String getINIT_CONSTANT_ALIAS() {
    return INIT_CONSTANT_ALIAS;
  }
  
  public String getLAYER_FIELD_INIT_VALUE() {
    return LAYER_FIELD_INIT_VALUE;
  }
  
  public String getINIT_ORTHOGONAL() {
    return INIT_ORTHOGONAL;
  }
  
  public String getINIT_ORTHOGONAL_ALIAS() {
    return INIT_ORTHOGONAL_ALIAS;
  }
  
  public String getLAYER_FIELD_INIT_GAIN() {
    return LAYER_FIELD_INIT_GAIN;
  }
  
  public String getINIT_TRUNCATED_NORMAL() {
    return INIT_TRUNCATED_NORMAL;
  }
  
  public String getINIT_TRUNCATED_NORMAL_ALIAS() {
    return INIT_TRUNCATED_NORMAL_ALIAS;
  }
  
  public String getINIT_IDENTITY() {
    return INIT_IDENTITY;
  }
  
  public String getINIT_IDENTITY_ALIAS() {
    return INIT_IDENTITY_ALIAS;
  }
  
  public String getINIT_VARIANCE_SCALING() {
    return INIT_VARIANCE_SCALING;
  }
  
  public String getLAYER_FIELD_INIT_MODE() {
    return LAYER_FIELD_INIT_MODE;
  }
  
  public String getLAYER_FIELD_INIT_DISTRIBUTION() {
    return LAYER_FIELD_INIT_DISTRIBUTION;
  }
  
  public String getLAYER_FIELD_POINT_WISE_INIT() {
    return LAYER_FIELD_POINT_WISE_INIT;
  }
  
  public String getLAYER_FIELD_W_REGULARIZER() {
    return LAYER_FIELD_W_REGULARIZER;
  }
  
  public String getLAYER_CLASS_NAME_INPUT() {
    return LAYER_CLASS_NAME_INPUT;
  }
  
  public String getKERAS_ACTIVATION_SOFTMAX() {
    return KERAS_ACTIVATION_SOFTMAX;
  }
  
  public String getKERAS_ACTIVATION_SOFTPLUS() {
    return KERAS_ACTIVATION_SOFTPLUS;
  }
  
  public String getKERAS_ACTIVATION_SOFTSIGN() {
    return KERAS_ACTIVATION_SOFTSIGN;
  }
  
  public String getKERAS_ACTIVATION_RELU() {
    return KERAS_ACTIVATION_RELU;
  }
  
  public String getKERAS_ACTIVATION_ELU() {
    return KERAS_ACTIVATION_ELU;
  }
  
  public String getKERAS_ACTIVATION_SELU() {
    return KERAS_ACTIVATION_SELU;
  }
  
  public String getKERAS_ACTIVATION_TANH() {
    return KERAS_ACTIVATION_TANH;
  }
  
  public String getKERAS_ACTIVATION_SIGMOID() {
    return KERAS_ACTIVATION_SIGMOID;
  }
  
  public String getKERAS_ACTIVATION_HARD_SIGMOID() {
    return KERAS_ACTIVATION_HARD_SIGMOID;
  }
  
  public String getKERAS_ACTIVATION_LINEAR() {
    return KERAS_ACTIVATION_LINEAR;
  }
  
  public String getLAYER_FIELD_ACTIVATION() {
    return LAYER_FIELD_ACTIVATION;
  }
  
  public String getLAYER_FIELD_B_REGULARIZER() {
    return LAYER_FIELD_B_REGULARIZER;
  }
  
  public String getLAYER_FIELD_NAME() {
    return LAYER_FIELD_NAME;
  }
  
  public String getLAYER_CLASS_NAME_TIME_DISTRIBUTED() {
    return LAYER_CLASS_NAME_TIME_DISTRIBUTED;
  }
  
  public String getLAYER_CLASS_NAME_ACTIVATION() {
    return LAYER_CLASS_NAME_ACTIVATION;
  }
  
  public String getLAYER_CLASS_NAME_LEAKY_RELU() {
    return LAYER_CLASS_NAME_LEAKY_RELU;
  }
  
  public String getLAYER_CLASS_NAME_DROPOUT() {
    return LAYER_CLASS_NAME_DROPOUT;
  }
  
  public String getLAYER_CLASS_NAME_ALPHA_DROPOUT() {
    return LAYER_CLASS_NAME_ALPHA_DROPOUT;
  }
  
  public String getLAYER_CLASS_NAME_GAUSSIAN_DROPOUT() {
    return LAYER_CLASS_NAME_GAUSSIAN_DROPOUT;
  }
  
  public String getLAYER_CLASS_NAME_GAUSSIAN_NOISE() {
    return LAYER_CLASS_NAME_GAUSSIAN_NOISE;
  }
  
  public String getLAYER_CLASS_NAME_DENSE() {
    return LAYER_CLASS_NAME_DENSE;
  }
  
  public String getLAYER_CLASS_NAME_TIME_DISTRIBUTED_DENSE() {
    return LAYER_CLASS_NAME_TIME_DISTRIBUTED_DENSE;
  }
  
  public String getLAYER_CLASS_NAME_BIDIRECTIONAL() {
    return LAYER_CLASS_NAME_BIDIRECTIONAL;
  }
  
  public String getLAYER_CLASS_NAME_LSTM() {
    return LAYER_CLASS_NAME_LSTM;
  }
  
  public String getLAYER_CLASS_NAME_SIMPLE_RNN() {
    return LAYER_CLASS_NAME_SIMPLE_RNN;
  }
  
  public String getLAYER_CLASS_NAME_CONVOLUTION_2D() {
    return LAYER_CLASS_NAME_CONVOLUTION_2D;
  }
  
  public String getLAYER_CLASS_NAME_DECONVOLUTION_2D() {
    return LAYER_CLASS_NAME_DECONVOLUTION_2D;
  }
  
  public String getLAYER_CLASS_NAME_CONVOLUTION_1D() {
    return LAYER_CLASS_NAME_CONVOLUTION_1D;
  }
  
  public String getLAYER_CLASS_NAME_ATROUS_CONVOLUTION_2D() {
    return LAYER_CLASS_NAME_ATROUS_CONVOLUTION_2D;
  }
  
  public String getLAYER_CLASS_NAME_ATROUS_CONVOLUTION_1D() {
    return LAYER_CLASS_NAME_ATROUS_CONVOLUTION_1D;
  }
  
  public String getLAYER_CLASS_NAME_SEPARABLE_CONVOLUTION_2D() {
    return LAYER_CLASS_NAME_SEPARABLE_CONVOLUTION_2D;
  }
  
  public String getLAYER_CLASS_NAME_MAX_POOLING_2D() {
    return LAYER_CLASS_NAME_MAX_POOLING_2D;
  }
  
  public String getLAYER_CLASS_NAME_AVERAGE_POOLING_2D() {
    return LAYER_CLASS_NAME_AVERAGE_POOLING_2D;
  }
  
  public String getLAYER_CLASS_NAME_MAX_POOLING_1D() {
    return LAYER_CLASS_NAME_MAX_POOLING_1D;
  }
  
  public String getLAYER_CLASS_NAME_AVERAGE_POOLING_1D() {
    return LAYER_CLASS_NAME_AVERAGE_POOLING_1D;
  }
  
  public String getLAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_1D() {
    return LAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_1D;
  }
  
  public String getLAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_2D() {
    return LAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_2D;
  }
  
  public String getLAYER_CLASS_NAME_GLOBAL_MAX_POOLING_2D() {
    return LAYER_CLASS_NAME_GLOBAL_MAX_POOLING_2D;
  }
  
  public String getLAYER_CLASS_NAME_GLOBAL_MAX_POOLING_1D() {
    return LAYER_CLASS_NAME_GLOBAL_MAX_POOLING_1D;
  }
  
  public String getLAYER_CLASS_NAME_BATCHNORMALIZATION() {
    return LAYER_CLASS_NAME_BATCHNORMALIZATION;
  }
  
  public String getLAYER_CLASS_NAME_EMBEDDING() {
    return LAYER_CLASS_NAME_EMBEDDING;
  }
  
  public String getLAYER_CLASS_NAME_MERGE() {
    return LAYER_CLASS_NAME_MERGE;
  }
  
  public String getLAYER_CLASS_NAME_FLATTEN() {
    return LAYER_CLASS_NAME_FLATTEN;
  }
  
  public String getLAYER_CLASS_NAME_RESHAPE() {
    return LAYER_CLASS_NAME_RESHAPE;
  }
  
  public String getLAYER_CLASS_NAME_ZERO_PADDING_2D() {
    return LAYER_CLASS_NAME_ZERO_PADDING_2D;
  }
  
  public String getLAYER_CLASS_NAME_UPSAMPLING_2D() {
    return LAYER_CLASS_NAME_UPSAMPLING_2D;
  }
  
  public String getLAYER_FIELD_CLASS_NAME() {
    return LAYER_FIELD_CLASS_NAME;
  }
  
  public String getLAYER_FIELD_CONFIG() {
    return LAYER_FIELD_CONFIG;
  }
  
  public String getLAYER_FIELD_LAYER() {
    return LAYER_FIELD_LAYER;
  }
  
  public String getLAYER_FIELD_BATCH_INPUT_SHAPE() {
    return LAYER_FIELD_BATCH_INPUT_SHAPE;
  }
  
  public String getLAYER_FIELD_BACKEND() {
    return LAYER_FIELD_BACKEND;
  }
  
  public String getLAYER_FIELD_DIM_ORDERING() {
    return LAYER_FIELD_DIM_ORDERING;
  }
  
  public String getDIM_ORDERING_TENSORFLOW() {
    return DIM_ORDERING_TENSORFLOW;
  }
  
  public String getDIM_ORDERING_THEANO() {
    return DIM_ORDERING_THEANO;
  }
  
  public String getLAYER_FIELD_INBOUND_NODES() {
    return LAYER_FIELD_INBOUND_NODES;
  }
  
  public String getLAYER_FIELD_OUTPUT_DIM() {
    return LAYER_FIELD_OUTPUT_DIM;
  }
  
  public String getLAYER_FIELD_EMBEDDING_OUTPUT_DIM() {
    return LAYER_FIELD_EMBEDDING_OUTPUT_DIM;
  }
  
  public String getLAYER_FIELD_NB_FILTER() {
    return LAYER_FIELD_NB_FILTER;
  }
  
  public String getLAYER_FIELD_DROPOUT() {
    return LAYER_FIELD_DROPOUT;
  }
  
  public String getLAYER_FIELD_DROPOUT_W() {
    return LAYER_FIELD_DROPOUT_W;
  }
  
  public String getLAYER_FIELD_USE_BIAS() {
    return LAYER_FIELD_USE_BIAS;
  }
  
  public String getLAYER_FIELD_MASK_ZERO() {
    return LAYER_FIELD_MASK_ZERO;
  }
  
  public String getKERAS_LOSS_MEAN_SQUARED_ERROR() {
    return KERAS_LOSS_MEAN_SQUARED_ERROR;
  }
  
  public String getKERAS_LOSS_MSE() {
    return KERAS_LOSS_MSE;
  }
  
  public String getKERAS_LOSS_MEAN_ABSOLUTE_ERROR() {
    return KERAS_LOSS_MEAN_ABSOLUTE_ERROR;
  }
  
  public String getKERAS_LOSS_MAE() {
    return KERAS_LOSS_MAE;
  }
  
  public String getKERAS_LOSS_MEAN_ABSOLUTE_PERCENTAGE_ERROR() {
    return KERAS_LOSS_MEAN_ABSOLUTE_PERCENTAGE_ERROR;
  }
  
  public String getKERAS_LOSS_MAPE() {
    return KERAS_LOSS_MAPE;
  }
  
  public String getKERAS_LOSS_MEAN_SQUARED_LOGARITHMIC_ERROR() {
    return KERAS_LOSS_MEAN_SQUARED_LOGARITHMIC_ERROR;
  }
  
  public String getKERAS_LOSS_MSLE() {
    return KERAS_LOSS_MSLE;
  }
  
  public String getKERAS_LOSS_SQUARED_HINGE() {
    return KERAS_LOSS_SQUARED_HINGE;
  }
  
  public String getKERAS_LOSS_HINGE() {
    return KERAS_LOSS_HINGE;
  }
  
  public String getKERAS_LOSS_SPARSE_CATEGORICAL_CROSSENTROPY() {
    return KERAS_LOSS_SPARSE_CATEGORICAL_CROSSENTROPY;
  }
  
  public String getKERAS_LOSS_BINARY_CROSSENTROPY() {
    return KERAS_LOSS_BINARY_CROSSENTROPY;
  }
  
  public String getKERAS_LOSS_CATEGORICAL_CROSSENTROPY() {
    return KERAS_LOSS_CATEGORICAL_CROSSENTROPY;
  }
  
  public String getKERAS_LOSS_KULLBACK_LEIBLER_DIVERGENCE() {
    return KERAS_LOSS_KULLBACK_LEIBLER_DIVERGENCE;
  }
  
  public String getKERAS_LOSS_KLD() {
    return KERAS_LOSS_KLD;
  }
  
  public String getKERAS_LOSS_POISSON() {
    return KERAS_LOSS_POISSON;
  }
  
  public String getKERAS_LOSS_COSINE_PROXIMITY() {
    return KERAS_LOSS_COSINE_PROXIMITY;
  }
  
  public String getLAYER_FIELD_CONVOLUTION_STRIDES() {
    return LAYER_FIELD_CONVOLUTION_STRIDES;
  }
  
  public String getLAYER_FIELD_SUBSAMPLE_LENGTH() {
    return LAYER_FIELD_SUBSAMPLE_LENGTH;
  }
  
  public String getLAYER_FIELD_POOL_STRIDES() {
    return LAYER_FIELD_POOL_STRIDES;
  }
  
  public String getLAYER_FIELD_POOL_1D_STRIDES() {
    return LAYER_FIELD_POOL_1D_STRIDES;
  }
  
  public String getLAYER_FIELD_DILATION_RATE() {
    return LAYER_FIELD_DILATION_RATE;
  }
  
  public String getLAYER_FIELD_UPSAMPLING_2D_SIZE() {
    return LAYER_FIELD_UPSAMPLING_2D_SIZE;
  }
  
  public String getLAYER_FIELD_UPSAMPLING_1D_SIZE() {
    return LAYER_FIELD_UPSAMPLING_1D_SIZE;
  }
  
  public String getLAYER_FIELD_NB_ROW() {
    return LAYER_FIELD_NB_ROW;
  }
  
  public String getLAYER_FIELD_NB_COL() {
    return LAYER_FIELD_NB_COL;
  }
  
  public String getLAYER_FIELD_FILTER_LENGTH() {
    return LAYER_FIELD_FILTER_LENGTH;
  }
  
  public String getLAYER_FIELD_POOL_SIZE() {
    return LAYER_FIELD_POOL_SIZE;
  }
  
  public String getLAYER_FIELD_POOL_1D_SIZE() {
    return LAYER_FIELD_POOL_1D_SIZE;
  }
  
  public String getLAYER_FIELD_KERNEL_SIZE() {
    return LAYER_FIELD_KERNEL_SIZE;
  }
  
  public String getLAYER_FIELD_BORDER_MODE() {
    return LAYER_FIELD_BORDER_MODE;
  }
  
  public String getLAYER_BORDER_MODE_SAME() {
    return LAYER_BORDER_MODE_SAME;
  }
  
  public String getLAYER_BORDER_MODE_VALID() {
    return LAYER_BORDER_MODE_VALID;
  }
  
  public String getLAYER_BORDER_MODE_FULL() {
    return LAYER_BORDER_MODE_FULL;
  }
  
  public String getLAYER_FIELD_ZERO_PADDING() {
    return LAYER_FIELD_ZERO_PADDING;
  }
}