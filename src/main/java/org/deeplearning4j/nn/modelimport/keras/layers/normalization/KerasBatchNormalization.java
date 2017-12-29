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

package org.deeplearning4j.nn.modelimport.keras.layers.normalization;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

//import org.deeplearning4j.nn.conf.layers.BatchNormalization;

/**
 * Imports a BatchNormalization layer from Keras.
 *
 * @author dave@skymind.io
 */
public class KerasBatchNormalization extends KerasLayer {
  private static final Logger log = LoggerFactory.getLogger(KerasInitilizationUtils.class);
  
  /* Keras layer configuration fields. */
  private final int LAYER_BATCHNORM_MODE_1 = 1;
  private final int LAYER_BATCHNORM_MODE_2 = 2;
  private final String LAYER_FIELD_GAMMA_REGULARIZER = "gamma_regularizer";
  private final String LAYER_FIELD_BETA_REGULARIZER = "beta_regularizer";
  private final String LAYER_FIELD_MODE = "mode";
  private final String LAYER_FIELD_AXIS = "axis";
  private final String LAYER_FIELD_MOMENTUM = "momentum";
  private final String LAYER_FIELD_EPSILON = "epsilon";
  
  /* Keras layer parameter names. */
  private final int NUM_TRAINABLE_PARAMS = 4;
  private final String PARAM_NAME_GAMMA = "gamma";
  private final String PARAM_NAME_BETA = "beta";
  private final String PARAM_NAME_RUNNING_MEAN = "running_mean";
  private final String PARAM_NAME_RUNNING_STD = "running_std";
  
  /**
   * Pass-through constructor from KerasLayer
   *
   * @param kerasVersion major keras version
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasBatchNormalization(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
    super(kerasVersion);
  }
  
  /**
   * Constructor from parsed Keras layer configuration dictionary.
   *
   * @param layerConfig dictionary containing Keras layer configuration
   * @throws InvalidKerasConfigurationException
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasBatchNormalization(Map<String, Object> layerConfig)
    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    this(layerConfig, true);
  }
  
  /**
   * Constructor from parsed Keras layer configuration dictionary.
   *
   * @param layerConfig           dictionary containing Keras layer configuration
   * @param enforceTrainingConfig whether to enforce training-related configuration options
   * @throws InvalidKerasConfigurationException
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasBatchNormalization(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    super(layerConfig, enforceTrainingConfig);
    getGammaRegularizerFromConfig(layerConfig, enforceTrainingConfig);
    getBetaRegularizerFromConfig(layerConfig, enforceTrainingConfig);
    int batchNormMode = getBatchNormMode(layerConfig, enforceTrainingConfig);
    if (batchNormMode != 0) {
      throw new UnsupportedKerasConfigurationException("Unsupported batch normalization mode " + batchNormMode +
        "Keras modes 1 and 2 have been removed from keras 2.x altogether." +
        "Try running with mode 0.");
    }
    int batchNormAxis = getBatchNormAxis(layerConfig, enforceTrainingConfig);
    if (!(batchNormAxis == 3 || batchNormAxis == -1)) {
      throw new UnsupportedKerasConfigurationException("Unsupported batch normalization axis " + batchNormAxis +
        "DL4J currently supports axis=3 only. Consider reshaping your input.");
    }
    throw new RuntimeException("NI");

//    BatchNormalization.Builder builder = new BatchNormalization.Builder().name(this.layerName).dropOut(this.dropout).minibatch(true)
//      .lockGammaBeta(false).eps(getEpsFromConfig(layerConfig));
//    this.layer = builder.build();
  }

//  /**
//   * Get DL4J BatchNormalizationLayer.
//   *
//   * @return BatchNormalizationLayer
//   */
//  public BatchNormalization getBatchNormalizationLayer() {
//    return (BatchNormalization) this.layer;
//  }
  
  /**
   * Get layer output type.
   *
   * @param inputType Array of InputTypes
   * @return output type as InputType
   * @throws InvalidKerasConfigurationException
   */
  public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException {
    if (inputType.length > 1) {
      throw new InvalidKerasConfigurationException(
        "Keras BatchNorm layer accepts only one input (received " + inputType.length + ")");
    }
    throw new RuntimeException("NI");
//    return this.getBatchNormalizationLayer().getOutputType(-1, inputType[0]);
  }
  
  /**
   * Returns number of trainable parameters in layer.
   *
   * @return number of trainable parameters (4)
   */
  @Override
  public int getNumParams() {
    return NUM_TRAINABLE_PARAMS;
  }
  
  /**
   * Set weights for layer.
   *
   * @param weights Map from parameter name to INDArray.
   */
  @Override
  public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
    this.weights = new HashMap<String, INDArray>();
    if (weights.containsKey(PARAM_NAME_BETA)) {
      this.weights.put(BatchNormalizationParamInitializer.BETA, weights.get(PARAM_NAME_BETA));
    }
    else {
      throw new InvalidKerasConfigurationException("Parameter " + PARAM_NAME_BETA + " does not exist in weights");
    }
    if (weights.containsKey(PARAM_NAME_GAMMA)) {
      this.weights.put(BatchNormalizationParamInitializer.GAMMA, weights.get(PARAM_NAME_GAMMA));
    }
    else {
      throw new InvalidKerasConfigurationException(
        "Parameter " + PARAM_NAME_GAMMA + " does not exist in weights");
    }
    if (weights.containsKey(conf.getLAYER_FIELD_BATCHNORMALIZATION_MOVING_MEAN())) {
      this.weights.put(BatchNormalizationParamInitializer.GLOBAL_MEAN, weights.get(conf.getLAYER_FIELD_BATCHNORMALIZATION_MOVING_MEAN()));
    }
    else {
      throw new InvalidKerasConfigurationException(
        "Parameter " + conf.getLAYER_FIELD_BATCHNORMALIZATION_MOVING_MEAN() + " does not exist in weights");
    }
    if (weights.containsKey(conf.getLAYER_FIELD_BATCHNORMALIZATION_MOVING_VARIANCE())) {
      this.weights.put(BatchNormalizationParamInitializer.GLOBAL_VAR, weights.get(conf.getLAYER_FIELD_BATCHNORMALIZATION_MOVING_VARIANCE()));
    }
    else {
      throw new InvalidKerasConfigurationException(
        "Parameter " + conf.getLAYER_FIELD_BATCHNORMALIZATION_MOVING_VARIANCE() + " does not exist in weights");
    }
    if (weights.size() > 4) {
      Set<String> paramNames = weights.keySet();
      paramNames.remove(PARAM_NAME_BETA);
      paramNames.remove(PARAM_NAME_GAMMA);
      paramNames.remove(conf.getLAYER_FIELD_BATCHNORMALIZATION_MOVING_MEAN());
      paramNames.remove(conf.getLAYER_FIELD_BATCHNORMALIZATION_MOVING_VARIANCE());
      String unknownParamNames = paramNames.toString();
      log.warn("Attemping to set weights for unknown parameters: "
        + unknownParamNames.substring(1, unknownParamNames.length() - 1));
    }
  }
  
  /**
   * Get BatchNormalization epsilon parameter from Keras layer configuration.
   *
   * @param layerConfig dictionary containing Keras layer configuration
   * @return epsilon
   * @throws InvalidKerasConfigurationException
   */
  protected double getEpsFromConfig(Map<String, Object> layerConfig) throws InvalidKerasConfigurationException {
    Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
    if (!innerConfig.containsKey(LAYER_FIELD_EPSILON)) {
      throw new InvalidKerasConfigurationException(
        "Keras BatchNorm layer config missing " + LAYER_FIELD_EPSILON + " field");
    }
    return (double) innerConfig.get(LAYER_FIELD_EPSILON);
  }
  
  /**
   * Get BatchNormalization momentum parameter from Keras layer configuration.
   *
   * @param layerConfig dictionary containing Keras layer configuration
   * @return momentum
   * @throws InvalidKerasConfigurationException
   */
  protected double getMomentumFromConfig(Map<String, Object> layerConfig) throws InvalidKerasConfigurationException {
    Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
    if (!innerConfig.containsKey(LAYER_FIELD_MOMENTUM)) {
      throw new InvalidKerasConfigurationException(
        "Keras BatchNorm layer config missing " + LAYER_FIELD_MOMENTUM + " field");
    }
    return (double) innerConfig.get(LAYER_FIELD_MOMENTUM);
  }
  
  /**
   * Get BatchNormalization gamma regularizer from Keras layer configuration. Currently unsupported.
   *
   * @param layerConfig dictionary containing Keras layer configuration
   * @return
   * @throws InvalidKerasConfigurationException
   */
  protected void getGammaRegularizerFromConfig(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
    throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
    Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
    if (innerConfig.get(LAYER_FIELD_GAMMA_REGULARIZER) != null) {
      if (enforceTrainingConfig) {
        throw new UnsupportedKerasConfigurationException(
          "Regularization for BatchNormalization gamma parameter not supported");
      }
      else {
        log.warn("Regularization for BatchNormalization gamma parameter not supported...ignoring.");
      }
    }
  }
  
  /**
   * Get BatchNormalization beta regularizer from Keras layer configuration. Currently unsupported.
   *
   * @param layerConfig dictionary containing Keras layer configuration
   * @return
   * @throws InvalidKerasConfigurationException
   */
  protected void getBetaRegularizerFromConfig(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
    throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
    Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
    if (innerConfig.get(LAYER_FIELD_BETA_REGULARIZER) != null) {
      if (enforceTrainingConfig) {
        throw new UnsupportedKerasConfigurationException(
          "Regularization for BatchNormalization beta parameter not supported");
      }
      else {
        log.warn("Regularization for BatchNormalization beta parameter not supported...ignoring.");
      }
    }
  }
  
  /**
   * Get BatchNormalization "mode" from Keras layer configuration. Most modes currently unsupported.
   *
   * @param layerConfig dictionary containing Keras layer configuration
   * @return
   * @throws InvalidKerasConfigurationException
   */
  protected int getBatchNormMode(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
    int batchNormMode = 0;
    if (this.kerasMajorVersion == 1 & !innerConfig.containsKey(LAYER_FIELD_MODE)) {
      throw new InvalidKerasConfigurationException(
        "Keras BatchNorm layer config missing " + LAYER_FIELD_MODE + " field");
    }
    if (this.kerasMajorVersion == 1) {
      batchNormMode = (int) innerConfig.get(LAYER_FIELD_MODE);
    }
    switch (batchNormMode) {
      case LAYER_BATCHNORM_MODE_1:
        throw new UnsupportedKerasConfigurationException("Keras BatchNormalization mode "
          + LAYER_BATCHNORM_MODE_1 + " (sample-wise) not supported");
      case LAYER_BATCHNORM_MODE_2:
        throw new UnsupportedKerasConfigurationException(
          "Keras BatchNormalization (per-batch statistics during testing) "
            + LAYER_BATCHNORM_MODE_2 + " not supported");
    }
    return batchNormMode;
  }
  
  /**
   * Get BatchNormalization axis from Keras layer configuration. Currently unused.
   *
   * @param layerConfig dictionary containing Keras layer configuration
   * @return
   * @throws InvalidKerasConfigurationException
   */
  protected int getBatchNormAxis(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
    throws InvalidKerasConfigurationException {
    Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
    return (int) innerConfig.get(LAYER_FIELD_AXIS);
  }
}
