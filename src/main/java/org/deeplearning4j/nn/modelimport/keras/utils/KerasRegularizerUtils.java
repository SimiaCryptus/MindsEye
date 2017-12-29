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

package org.deeplearning4j.nn.modelimport.keras.utils;

import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import java.util.Map;

public class KerasRegularizerUtils {
  
  /**
   * Get weight regularization from Keras weight regularization configuration.
   *
   * @param layerConfig     Map containing Keras weight regularization configuration
   * @param conf            Keras layer configuration
   * @param configField     regularization config field to use
   * @param regularizerType type of regularization as string (e.g. "l2")
   * @return L1 or L2 regularization strength (0.0 if none)
   */
  public static double getWeightRegularizerFromConfig(Map<String, Object> layerConfig,
                                                      KerasLayerConfiguration conf,
                                                      String configField,
                                                      String regularizerType)
    throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
    Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
    if (innerConfig.containsKey(configField)) {
      Map<String, Object> regularizerConfig = (Map<String, Object>) innerConfig.get(configField);
      if (regularizerConfig != null && regularizerConfig.containsKey(regularizerType)) {
        return (double) regularizerConfig.get(regularizerType);
      }
    }
    return 0.0;
  }
}
