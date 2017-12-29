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

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.primitives.Pair;

import java.util.Map;

import static org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolutionUtils.getDilationRate;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils.getWeightInitFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.getHasBiasFromConfig;


/**
 * Imports a 2D Convolution layer from Keras.
 *
 * @author dave@skymind.io
 */
public class KerasConvolution2D extends KerasConvolution {
  
  /**
   * Pass-through constructor from KerasLayer
   *
   * @param kerasVersion major keras version
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasConvolution2D(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
    super(kerasVersion);
  }
  
  /**
   * Constructor from parsed Keras layer configuration dictionary.
   *
   * @param layerConfig dictionary containing Keras layer configuration
   * @throws InvalidKerasConfigurationException
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasConvolution2D(Map<String, Object> layerConfig)
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
  public KerasConvolution2D(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    super(layerConfig, enforceTrainingConfig);
    
    hasBias = getHasBiasFromConfig(layerConfig, conf);
    numTrainableParams = hasBias ? 2 : 1;
    int[] dilationRate = getDilationRate(layerConfig, 2, conf, false);
    
    Pair<WeightInit, Distribution> init = getWeightInitFromConfig(layerConfig, conf.getLAYER_FIELD_INIT(),
      enforceTrainingConfig, conf, kerasMajorVersion);
    WeightInit weightInit = init.getFirst();
    Distribution distribution = init.getSecond();
    
    throw new RuntimeException("NI");
//        ConvolutionLayer.Builder builder = new ConvolutionLayer.Builder().name(this.layerName)
//                .nOut(getNOutFromConfig(layerConfig, conf)).dropOut(this.dropout)
//                .activation(getActivationFromConfig(layerConfig, conf))
//                .weightInit(weightInit)
//                .l1(this.weightL1Regularization).l2(this.weightL2Regularization)
//                .convolutionMode(getConvolutionModeFromConfig(layerConfig, conf))
//                .kernelSize(getKernelSizeFromConfig(layerConfig, 2, conf, kerasMajorVersion))
//                .hasBias(hasBias)
//                .stride(getStrideFromConfig(layerConfig, 2, conf));
//        int[] padding = getPaddingFromBorderModeConfig(layerConfig, 2, conf, kerasMajorVersion);
//        if (distribution != null)
//            builder.dist(distribution);
//        if (hasBias)
//            builder.biasInit(0.0);
//        if (padding != null)
//            builder.padding(padding);
//        if (dilationRate != null)
//            builder.dilation(dilationRate);
//        this.layer = builder.build();
  }
  
  /**
   * Get DL4J ConvolutionLayer.
   *
   * @return ConvolutionLayer
   */
  public ConvolutionLayer getConvolution2DLayer() {
    return (ConvolutionLayer) this.layer;
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
      return this.getConvolution2DLayer().getOutputType(-1, preprocessor.getOutputType(inputType[0]));
    }
    return this.getConvolution2DLayer().getOutputType(-1, inputType[0]);
  }
  
}
