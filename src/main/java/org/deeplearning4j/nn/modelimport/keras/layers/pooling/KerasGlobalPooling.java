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
package org.deeplearning4j.nn.modelimport.keras.layers.pooling;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import java.util.Map;

import static org.deeplearning4j.nn.modelimport.keras.layers.pooling.KerasPoolingUtils.mapPoolingDimensions;

/**
 * Imports a Keras Pooling layer as a DL4J Subsampling layer.
 *
 * @author dave@skymind.io
 */
public class KerasGlobalPooling extends KerasLayer {
  
  private final int[] dimensions;
  
  /**
   * Constructor from parsed Keras layer configuration dictionary.
   *
   * @param layerConfig dictionary containing Keras layer configuration.
   * @throws InvalidKerasConfigurationException
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasGlobalPooling(Map<String, Object> layerConfig)
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
  public KerasGlobalPooling(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    super(layerConfig, enforceTrainingConfig);
    this.dimensions = mapPoolingDimensions(this.className, conf);
//        GlobalPoolingLayer.Builder builder =
//                        new GlobalPoolingLayer.Builder(mapPoolingType(this.className, conf))
//                                .poolingDimensions(dimensions)
//                                .collapseDimensions(false) // while this is true by default in dl4j, it's false in keras.
//                                .name(this.layerName)
//                                .dropOut(this.dropout);
//        this.layer = builder.build();
//        this.vertex = null;
  }

//    /**
//     * Get DL4J SubsamplingLayer.
//     *
//     * @return  SubsamplingLayer
//     */
//    public GlobalPoolingLayer getGlobalPoolingLayer() {
//        return (GlobalPoolingLayer) this.layer;
//    }
  
  /**
   * Gets appropriate DL4J InputPreProcessor for given InputTypes.
   *
   * @param inputType Array of InputTypes
   * @return DL4J InputPreProcessor
   * @throws InvalidKerasConfigurationException
   * @see InputPreProcessor
   */
  public InputPreProcessor getInputPreprocessor(InputType... inputType) throws InvalidKerasConfigurationException {
      if (inputType.length > 1) {
          throw new InvalidKerasConfigurationException(
            "Keras GlobalPooling layer accepts only one input (received " + inputType.length + ")");
      }
    InputPreProcessor preprocessor;
    throw new RuntimeException("NI");
//        if (inputType[0].getType() == InputType.Type.FF && this.dimensions.length == 1) {
//            preprocessor = new FeedForwardToRnnPreProcessor();
//        } else {
//            preprocessor = this.getGlobalPoolingLayer().getPreProcessorForInputType(inputType[0]);
//        }
//        return preprocessor;
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
            "Keras Subsampling layer accepts only one input (received " + inputType.length + ")");
      }

        /* Check whether layer requires a preprocessor for this InputType. */
    throw new RuntimeException("NI");
//        InputPreProcessor preprocessor = getInputPreprocessor(inputType[0]);
//        if (preprocessor != null) {
//            return this.getGlobalPoolingLayer().getOutputType(-1, preprocessor.getOutputType(inputType[0]));
//        }
//        return this.getGlobalPoolingLayer().getOutputType(-1, inputType[0]);
  }
}
