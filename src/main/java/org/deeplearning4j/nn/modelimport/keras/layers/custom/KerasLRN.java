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

package org.deeplearning4j.nn.modelimport.keras.layers.custom;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;

import java.util.Map;

/**
 * Keras does not have an official LRN layer. Instead, the Keras community has
 * developed helpers to help address this issue. This custom layer was built specifically
 * to allow import of GoogLeNet https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14.
 *
 * @author Justin Long (crockpotveggies)
 */
public class KerasLRN extends KerasLayer {
  
  /**
   * Constructor from parsed Keras layer configuration dictionary.
   *
   * @param layerConfig dictionary containing Keras layer configuration.
   * @throws InvalidKerasConfigurationException
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasLRN(Map<String, Object> layerConfig)
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
  public KerasLRN(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    super(layerConfig, enforceTrainingConfig);
    Map<String, Object> lrnParams = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
    
    LocalResponseNormalization.Builder builder = new LocalResponseNormalization.Builder().name(this.layerName)
      .dropOut(this.dropout).alpha((double) lrnParams.get("alpha"))
      .beta((double) lrnParams.get("beta")).k((int) lrnParams.get("k")).n((int) lrnParams.get("n"));
    this.layer = builder.build();
    this.vertex = null;
  }
  
  /**
   * Get DL4J LRN.
   *
   * @return LocalResponseNormalization
   */
  public LocalResponseNormalization getLocalResponseNormalization() {
    return (LocalResponseNormalization) this.layer;
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
            "Keras LRN layer accepts only one input (received " + inputType.length + ")");
      }
    return this.getLocalResponseNormalization().getOutputType(-1, inputType[0]);
  }
}
