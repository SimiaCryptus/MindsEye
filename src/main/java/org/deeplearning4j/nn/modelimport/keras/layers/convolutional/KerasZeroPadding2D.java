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

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import java.util.Map;

import static org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolutionUtils.getZeroPaddingFromConfig;

/**
 * Imports a Keras ZeroPadding 2D layer.
 *
 * @author dave@skymind.io
 */
public class KerasZeroPadding2D extends KerasLayer {
  
  /**
   * Constructor from parsed Keras layer configuration dictionary.
   *
   * @param layerConfig dictionary containing Keras layer configuration.
   * @throws InvalidKerasConfigurationException
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasZeroPadding2D(Map<String, Object> layerConfig)
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
  public KerasZeroPadding2D(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    super(layerConfig, enforceTrainingConfig);
    ZeroPaddingLayer.Builder builder = new ZeroPaddingLayer.Builder(
      getZeroPaddingFromConfig(layerConfig, conf, 2))
      .name(this.layerName).dropOut(this.dropout);
    this.layer = builder.build();
    this.vertex = null;
  }
  
  /**
   * Get DL4J SubsamplingLayer.
   *
   * @return SubsamplingLayer
   */
  public ZeroPaddingLayer getZeroPadding2DLayer() {
    return (ZeroPaddingLayer) this.layer;
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
        "Keras ZeroPadding layer accepts only one input (received " + inputType.length + ")");
    }
    return this.getZeroPadding2DLayer().getOutputType(-1, inputType[0]);
  }
}
