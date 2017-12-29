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

import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;

import java.util.Map;

/**
 * Utility functions for Keras RNN layers
 *
 * @author Max Pumperla
 */
public class KerasRnnUtils {
  
  /**
   * Get unroll parameter to decide whether to unroll RNN with BPTT or not.
   *
   * @param conf        KerasLayerConfiguration
   * @param layerConfig dictionary containing Keras layer properties
   * @return boolean unroll parameter
   * @throws InvalidKerasConfigurationException Invalid Keras configuration
   */
  public static boolean getUnrollRecurrentLayer(KerasLayerConfiguration conf, Map<String, Object> layerConfig)
    throws InvalidKerasConfigurationException {
    Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
    if (!innerConfig.containsKey(conf.getLAYER_FIELD_UNROLL())) {
      throw new InvalidKerasConfigurationException(
        "Keras LSTM layer config missing " + conf.getLAYER_FIELD_UNROLL() + " field");
    }
    return (boolean) innerConfig.get(conf.getLAYER_FIELD_UNROLL());
  }
  
  /**
   * Get recurrent weight dropout from Keras layer configuration.
   * Non-zero dropout rates are currently not supported.
   *
   * @param conf        KerasLayerConfiguration
   * @param layerConfig dictionary containing Keras layer properties
   * @return recurrent dropout rate
   * @throws InvalidKerasConfigurationException Invalid Keras configuration
   */
  public static double getRecurrentDropout(KerasLayerConfiguration conf, Map<String, Object> layerConfig)
    throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
    Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
    double dropout = 1.0;
    if (innerConfig.containsKey(conf.getLAYER_FIELD_DROPOUT_U())) {
      dropout = 1.0 - (double) innerConfig.get(conf.getLAYER_FIELD_DROPOUT_U());
    }
    if (dropout < 1.0) {
      throw new UnsupportedKerasConfigurationException(
        "Dropout > 0 on recurrent connections not supported.");
    }
    return dropout;
  }
}
