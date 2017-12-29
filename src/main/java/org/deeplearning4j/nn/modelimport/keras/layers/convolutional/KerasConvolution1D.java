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
package org.deeplearning4j.nn.modelimport.keras.layers.convolutional;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution1DLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolutionUtils.getDilationRate;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils.getWeightInitFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.getHasBiasFromConfig;

/**
 * Imports a 1D Convolution layer from Keras.
 *
 * @author Max Pumperla
 */
public class KerasConvolution1D extends KerasConvolution {
  private static final Logger log = LoggerFactory.getLogger(KerasInitilizationUtils.class);
  
  /**
   * Pass-through constructor from KerasLayer
   *
   * @param kerasVersion major keras version
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasConvolution1D(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
    super(kerasVersion);
  }
  
  /**
   * Constructor from parsed Keras layer configuration dictionary.
   *
   * @param layerConfig dictionary containing Keras layer configuration
   * @throws InvalidKerasConfigurationException
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasConvolution1D(Map<String, Object> layerConfig)
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
  public KerasConvolution1D(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    super(layerConfig, enforceTrainingConfig);
    hasBias = getHasBiasFromConfig(layerConfig, conf);
    numTrainableParams = hasBias ? 2 : 1;
    int[] dilationRate = getDilationRate(layerConfig, 1, conf, false);
    
    Pair<WeightInit, Distribution> init = getWeightInitFromConfig(layerConfig, conf.getLAYER_FIELD_INIT(),
      enforceTrainingConfig, conf, kerasMajorVersion);
    WeightInit weightInit = init.getFirst();
    Distribution distribution = init.getSecond();
    
    throw new RuntimeException("NI");
//        Convolution1DLayer.Builder builder = new Convolution1DLayer.Builder().name(this.layerName)
//                .nOut(getNOutFromConfig(layerConfig, conf)).dropOut(this.dropout)
//                .activation(getActivationFromConfig(layerConfig, conf))
//                .weightInit(weightInit)
//                .l1(this.weightL1Regularization).l2(this.weightL2Regularization)
//                .convolutionMode(getConvolutionModeFromConfig(layerConfig, conf))
//                .kernelSize(getKernelSizeFromConfig(layerConfig, 1,  conf, kerasMajorVersion)[0])
//                .hasBias(hasBias)
//                .stride(getStrideFromConfig(layerConfig, 1, conf)[0]);
//        int[] padding = getPaddingFromBorderModeConfig(layerConfig, 1, conf, kerasMajorVersion);
//        if (distribution != null)
//            builder.dist(distribution);
//        if (hasBias)
//            builder.biasInit(0.0);
//        if (padding != null)
//            builder.padding(padding[0]);
//        if (dilationRate != null)
//            builder.dilation(dilationRate);
//        this.layer = builder.build();
  }
  
  /**
   * Get DL4J ConvolutionLayer.
   *
   * @return ConvolutionLayer
   */
  public Convolution1DLayer getConvolution1DLayer() {
    return (Convolution1DLayer) this.layer;
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
            "Keras Convolution layer accepts only one input (received " + inputType.length + ")");
      }
    InputPreProcessor preprocessor = getInputPreprocessor(inputType[0]);
    if (preprocessor != null) {
      return this.getConvolution1DLayer().getOutputType(-1, preprocessor.getOutputType(inputType[0]));
    }
    return this.getConvolution1DLayer().getOutputType(-1, inputType[0]);
  }
  
  /**
   * Set weights for layer.
   *
   * @param weights Map from parameter name to INDArray.
   */
  @Override
  public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
    this.weights = new HashMap<>();
      if (weights.containsKey(conf.getKERAS_PARAM_NAME_W())) {
          INDArray kerasParamValue = weights.get(conf.getKERAS_PARAM_NAME_W());
          INDArray paramValue;
          switch (this.getDimOrder()) {
              case TENSORFLOW:
                  paramValue = kerasParamValue.permute(2, 1, 0);
                  paramValue = paramValue.reshape(
                    paramValue.size(0), paramValue.size(1), paramValue.size(2), 1);
                  break;
              case THEANO:
                  paramValue = kerasParamValue.reshape(
                    kerasParamValue.size(0), kerasParamValue.size(1),
                    kerasParamValue.size(2), 1).dup();
                  for (int i = 0; i < paramValue.tensorssAlongDimension(2, 3); i++) {
                      INDArray copyFilter = paramValue.tensorAlongDimension(i, 2, 3).dup();
                      double[] flattenedFilter = copyFilter.ravel().data().asDouble();
                      ArrayUtils.reverse(flattenedFilter);
                      INDArray newFilter = Nd4j.create(flattenedFilter, copyFilter.shape());
                      INDArray inPlaceFilter = paramValue.tensorAlongDimension(i, 2, 3);
                      inPlaceFilter.muli(0).addi(newFilter);
                  }
                  break;
              default:
                  throw new InvalidKerasConfigurationException("Unknown keras backend " + this.getDimOrder());
          }
          this.weights.put(ConvolutionParamInitializer.WEIGHT_KEY, paramValue);
      }
      else {
          throw new InvalidKerasConfigurationException(
            "Parameter " + conf.getKERAS_PARAM_NAME_W() + " does not exist in weights");
      }
    
    if (hasBias) {
        if (weights.containsKey(conf.getKERAS_PARAM_NAME_B())) {
            this.weights.put(ConvolutionParamInitializer.BIAS_KEY, weights.get(conf.getKERAS_PARAM_NAME_B()));
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
