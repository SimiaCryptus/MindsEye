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

package org.deeplearning4j.nn.modelimport.keras.layers.core;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils.getWeightInitFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.getHasBiasFromConfig;

/**
 * Imports a Dense layer from Keras.
 *
 * @author dave@skymind.io
 */
public class KerasDense extends KerasLayer {
  private static final Logger log = LoggerFactory.getLogger(KerasInitilizationUtils.class);
  
  /* Keras layer parameter names. */
  private int numTrainableParams = 2;
  private boolean hasBias;
  
  /**
   * Pass-through constructor from KerasLayer
   *
   * @param kerasVersion major keras version
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasDense(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
    super(kerasVersion);
  }
  
  /**
   * Constructor from parsed Keras layer configuration dictionary.
   *
   * @param layerConfig dictionary containing Keras layer configuration
   * @throws InvalidKerasConfigurationException
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasDense(Map<String, Object> layerConfig)
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
  public KerasDense(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    super(layerConfig, enforceTrainingConfig);
    hasBias = getHasBiasFromConfig(layerConfig, conf);
    numTrainableParams = hasBias ? 2 : 1;
    
    Pair<WeightInit, Distribution> init = getWeightInitFromConfig(layerConfig, conf.getLAYER_FIELD_INIT(),
      enforceTrainingConfig, conf, kerasMajorVersion);
    WeightInit weightInit = init.getFirst();
    Distribution distribution = init.getSecond();
    
    throw new RuntimeException("NI");
//        DenseLayer.Builder builder = new DenseLayer.Builder().name(this.layerName).nOut(getNOutFromConfig(layerConfig, conf))
//                .dropOut(this.dropout).activation(getActivationFromConfig(layerConfig, conf))
//                .weightInit(weightInit)
//                .biasInit(0.0)
//                .l1(this.weightL1Regularization).l2(this.weightL2Regularization)
//                .hasBias(hasBias);
//        if (distribution != null)
//            builder.dist(distribution);
//        this.layer = builder.build();
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
        /* Check whether layer requires a preprocessor for this InputType. */
    InputPreProcessor preprocessor = getInputPreprocessor(inputType[0]);
    throw new RuntimeException("NI");
//    if (preprocessor != null) {
//      return this.getDenseLayer().getOutputType(-1, preprocessor.getOutputType(inputType[0]));
//    }
//    return this.getDenseLayer().getOutputType(-1, inputType[0]);
  }
  
  /**
   * Returns number of trainable parameters in layer.
   *
   * @return number of trainable parameters (2)
   */
  @Override
  public int getNumParams() {
    return numTrainableParams;
  }
  
  /**
   * Set weights for layer.
   *
   * @param weights
   */
  @Override
  public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
    this.weights = new HashMap<String, INDArray>();
    if (weights.containsKey(conf.getKERAS_PARAM_NAME_W())) {
      this.weights.put(DefaultParamInitializer.WEIGHT_KEY, weights.get(conf.getKERAS_PARAM_NAME_W()));
    }
    else {
      throw new InvalidKerasConfigurationException(
        "Parameter " + conf.getKERAS_PARAM_NAME_W() + " does not exist in weights");
    }
    if (hasBias) {
      if (weights.containsKey(conf.getKERAS_PARAM_NAME_B())) {
        this.weights.put(DefaultParamInitializer.BIAS_KEY, weights.get(conf.getKERAS_PARAM_NAME_B()));
      }
      else {
        throw new InvalidKerasConfigurationException(
          "Parameter " + conf.getKERAS_PARAM_NAME_B() + " does not exist in weights");
      }
    }
    if (weights.size() > 2) {
      Set<String> paramNames = weights.keySet();
      paramNames.remove(conf.getKERAS_PARAM_NAME_W());
      paramNames.remove(conf.getKERAS_PARAM_NAME_B());
      String unknownParamNames = paramNames.toString();
      log.warn("Attemping to set weights for unknown parameters: "
        + unknownParamNames.substring(1, unknownParamNames.length() - 1));
    }
  }
}
