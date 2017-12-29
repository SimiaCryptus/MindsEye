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

package org.deeplearning4j.nn.modelimport.keras.layers.noise;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;

import java.util.Map;

public class KerasGaussianNoise extends KerasLayer {
  
  /**
   * Pass-through constructor from KerasLayer
   *
   * @param kerasVersion major keras version
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasGaussianNoise(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
    super(kerasVersion);
  }
  
  /**
   * Constructor from parsed Keras layer configuration dictionary.
   *
   * @param layerConfig dictionary containing Keras layer configuration.
   * @throws InvalidKerasConfigurationException
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasGaussianNoise(Map<String, Object> layerConfig)
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
  public KerasGaussianNoise(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    super(layerConfig, enforceTrainingConfig);
    Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
    if (!innerConfig.containsKey(conf.getLAYER_FIELD_GAUSSIAN_VARIANCE())) {
      throw new InvalidKerasConfigurationException("Keras configuration does not contain "
        + conf.getLAYER_FIELD_GAUSSIAN_VARIANCE() + " parameter" +
        "needed for GaussianNoise");
    }
    double stddev = (double) innerConfig.get(conf.getLAYER_FIELD_GAUSSIAN_VARIANCE());
    
    throw new RuntimeException("NI");
//        this.layer = new DropoutLayer.Builder().name(this.layerName)
//                .dropOut(new GaussianNoise(stddev)).build();
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
        "Keras Gaussian Noise layer accepts only one input (received " + inputType.length + ")");
    }
    return this.getGaussianNoiseLayer().getOutputType(-1, inputType[0]);
  }
  
  /**
   * Get DL4J DropoutLayer with Gaussian dropout.
   *
   * @return DropoutLayer
   */
  public DropoutLayer getGaussianNoiseLayer() {
    return (DropoutLayer) this.layer;
  }
}
