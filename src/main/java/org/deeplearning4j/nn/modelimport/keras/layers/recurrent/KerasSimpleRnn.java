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

package org.deeplearning4j.nn.modelimport.keras.layers.recurrent;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.InputTypeUtil;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.deeplearning4j.nn.params.SimpleRnnParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils.getWeightInitFromConfig;

/**
 * Imports a Keras SimpleRNN layer as a DL4J SimpleRnn layer.
 *
 * @author Max Pumperla
 */
public class KerasSimpleRnn extends KerasLayer {
  private static final Logger log = LoggerFactory.getLogger(KerasInitilizationUtils.class);
  
  private final int NUM_TRAINABLE_PARAMS = 3;
  protected boolean unroll = false;
  protected boolean returnSequences;
  
  /**
   * Pass-through constructor from KerasLayer
   *
   * @param kerasVersion major keras version
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasSimpleRnn(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
    super(kerasVersion);
  }
  
  /**
   * Constructor from parsed Keras layer configuration dictionary.
   *
   * @param layerConfig dictionary containing Keras layer configuration.
   * @throws InvalidKerasConfigurationException
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasSimpleRnn(Map<String, Object> layerConfig)
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
  public KerasSimpleRnn(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    super(layerConfig, enforceTrainingConfig);
    
    Pair<WeightInit, Distribution> init = getWeightInitFromConfig(layerConfig, conf.getLAYER_FIELD_INIT(),
      enforceTrainingConfig, conf, kerasMajorVersion);
    WeightInit weightInit = init.getFirst();
    Distribution distribution = init.getSecond();
    
    Pair<WeightInit, Distribution> recurrentInit = getWeightInitFromConfig(layerConfig, conf.getLAYER_FIELD_INNER_INIT(),
      enforceTrainingConfig, conf, kerasMajorVersion);
    WeightInit recurrentWeightInit = recurrentInit.getFirst();
    Distribution recurrentDistribution = recurrentInit.getSecond();
    
    Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
    this.returnSequences = (Boolean) innerConfig.get(conf.getLAYER_FIELD_RETURN_SEQUENCES());
    
      if (weightInit != recurrentWeightInit || distribution != recurrentDistribution) {
          if (enforceTrainingConfig) {
              throw new UnsupportedKerasConfigurationException(
                "Specifying different initialization for recurrent weights not supported.");
          }
          else {
              log.warn("Specifying different initialization for recurrent weights not supported.");
          }
      }
    KerasRnnUtils.getRecurrentDropout(conf, layerConfig);
    this.unroll = KerasRnnUtils.getUnrollRecurrentLayer(conf, layerConfig);
    
    throw new RuntimeException("NI");
//        SimpleRnn.Builder builder = new SimpleRnn.Builder()
//                .name(this.layerName)
//                .nOut(getNOutFromConfig(layerConfig, conf))
//                .dropOut(this.dropout)
//                .activation(getActivationFromConfig(layerConfig, conf))
//                .weightInit(weightInit)
//                .biasInit(0.0)
//                .l1(this.weightL1Regularization)
//                .l2(this.weightL2Regularization);
//        if (distribution != null)
//            builder.dist(distribution);
//        if (this.returnSequences)
//            this.layer = builder.build();
//        else
//        {
//            throw new RuntimeException("NI");
////            this.layer = new LastTimeStep(builder.build());
//        }
  }
  
  /**
   * Get DL4J SimpleRnn layer.
   *
   * @return SimpleRnn Layer
   */
  public Layer getSimpleRnnLayer() {
    return this.layer;
  }
  
  /**
   * Get layer output type.
   *
   * @param inputType Array of InputTypes
   * @return output type as InputType
   * @throws InvalidKerasConfigurationException
   */
  @Override
  public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException {
      if (inputType.length > 1) {
          throw new InvalidKerasConfigurationException(
            "Keras SimpleRnn layer accepts only one input (received " + inputType.length + ")");
      }
    InputPreProcessor preProcessor = getInputPreprocessor(inputType);
      if (preProcessor != null) {
          return preProcessor.getOutputType(inputType[0]);
      }
      else {
          return this.getSimpleRnnLayer().getOutputType(-1, inputType[0]);
      }
  }
  
  /**
   * Returns number of trainable parameters in layer.
   *
   * @return number of trainable parameters (12)
   */
  @Override
  public int getNumParams() {
    return NUM_TRAINABLE_PARAMS;
  }
  
  /**
   * Gets appropriate DL4J InputPreProcessor for given InputTypes.
   *
   * @param inputType Array of InputTypes
   * @return DL4J InputPreProcessor
   * @throws InvalidKerasConfigurationException Invalid Keras configuration exception
   * @see InputPreProcessor
   */
  @Override
  public InputPreProcessor getInputPreprocessor(InputType... inputType) throws InvalidKerasConfigurationException {
      if (inputType.length > 1) {
          throw new InvalidKerasConfigurationException(
            "Keras SimpleRnn layer accepts only one input (received " + inputType.length + ")");
      }
    
    return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType[0], layerName);
  }
  
  /**
   * Get whether SimpleRnn layer should be unrolled (for truncated BPTT).
   *
   * @return
   */
  public boolean getUnroll() {
    return this.unroll;
  }
  
  
  /**
   * Set weights for layer.
   *
   * @param weights
   */
  @Override
  public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
    this.weights = new HashMap<>();
    
    
    INDArray W;
      if (weights.containsKey(conf.getKERAS_PARAM_NAME_W())) {
          W = weights.get(conf.getKERAS_PARAM_NAME_W());
      }
      else {
          throw new InvalidKerasConfigurationException(
            "Keras SimpleRNN layer does not contain parameter " + conf.getKERAS_PARAM_NAME_W());
      }
    this.weights.put(SimpleRnnParamInitializer.WEIGHT_KEY, W);
    
    
    INDArray RW;
      if (weights.containsKey(conf.getKERAS_PARAM_NAME_RW())) {
          RW = weights.get(conf.getKERAS_PARAM_NAME_RW());
      }
      else {
          throw new InvalidKerasConfigurationException(
            "Keras SimpleRNN layer does not contain parameter " + conf.getKERAS_PARAM_NAME_RW());
      }
    this.weights.put(SimpleRnnParamInitializer.RECURRENT_WEIGHT_KEY, RW);
    
    
    INDArray b;
      if (weights.containsKey(conf.getKERAS_PARAM_NAME_B())) {
          b = weights.get(conf.getKERAS_PARAM_NAME_B());
      }
      else {
          throw new InvalidKerasConfigurationException(
            "Keras SimpleRNN layer does not contain parameter " + conf.getKERAS_PARAM_NAME_B());
      }
    this.weights.put(SimpleRnnParamInitializer.BIAS_KEY, b);
    
    
    if (weights.size() > NUM_TRAINABLE_PARAMS) {
      Set<String> paramNames = weights.keySet();
      paramNames.remove(conf.getKERAS_PARAM_NAME_B());
      paramNames.remove(conf.getKERAS_PARAM_NAME_W());
      paramNames.remove(conf.getKERAS_PARAM_NAME_RW());
      String unknownParamNames = paramNames.toString();
      log.warn("Attemping to set weights for unknown parameters: "
        + unknownParamNames.substring(1, unknownParamNames.length() - 1));
    }
  }
  
}
