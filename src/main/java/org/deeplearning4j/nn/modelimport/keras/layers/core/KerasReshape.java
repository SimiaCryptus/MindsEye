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
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.ReshapePreprocessor;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.List;
import java.util.Map;

/**
 * Imports Reshape layer from Keras
 *
 * @author Max Pumperla
 */
public class KerasReshape extends KerasLayer {
  
  private int[] targetShape;
  
  
  /**
   * Constructor from parsed Keras layer configuration dictionary.
   *
   * @param layerConfig dictionary containing Keras layer configuration
   * @throws InvalidKerasConfigurationException
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasReshape(Map<String, Object> layerConfig)
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
  public KerasReshape(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    super(layerConfig, enforceTrainingConfig);
    Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
    String targetShape = "target_shape";
    if (innerConfig.containsKey(targetShape)) {
      List<Integer> targetShapeList = (List<Integer>) innerConfig.get(targetShape);
      this.targetShape = ArrayUtil.toArray(targetShapeList);
    }
    
  }
  
  /**
   * Whether this Keras layer maps to a DL4J InputPreProcessor.
   *
   * @return true
   */
  @Override
  public boolean isInputPreProcessor() {
    return true;
  }
  
  /**
   * Gets appropriate DL4J InputPreProcessor for given InputTypes.
   *
   * @param inputType Array of InputTypes
   * @return DL4J InputPreProcessor
   * @throws InvalidKerasConfigurationException
   * @see InputPreProcessor
   */
  @Override
  public InputPreProcessor getInputPreprocessor(InputType... inputType) throws InvalidKerasConfigurationException {
      if (inputType.length > 1) {
          throw new InvalidKerasConfigurationException(
            "Keras Reshape layer accepts only one input (received " + inputType.length + ")");
      }
    InputPreProcessor preprocessor = null;
    if (inputType[0] instanceof InputType.InputTypeConvolutional) {
      InputType.InputTypeConvolutional it = (InputType.InputTypeConvolutional) inputType[0];
      switch (this.getDimOrder()) {
        case NONE:
        case THEANO:
          int[] inputShapeTh = {it.getHeight(), it.getWidth(), it.getDepth()};
          preprocessor = new ReshapePreprocessor(inputShapeTh, this.targetShape);
          break;
        case TENSORFLOW:
          int[] inputShapeTf = {it.getWidth(), it.getDepth(), it.getHeight()};
          preprocessor = new ReshapePreprocessor(inputShapeTf, this.targetShape);
        
      }
    }
    else if (inputType[0] instanceof InputType.InputTypeRecurrent) {
      InputType.InputTypeRecurrent it = (InputType.InputTypeRecurrent) inputType[0];
      int[] inputShape = {it.getSize(), it.getTimeSeriesLength()};
      preprocessor = new ReshapePreprocessor(inputShape, this.targetShape);
    }
    else if (inputType[0] instanceof InputType.InputTypeFeedForward) {
      InputType.InputTypeFeedForward it = (InputType.InputTypeFeedForward) inputType[0];
      int[] inputShape = {it.getSize()};
      preprocessor = new ReshapePreprocessor(inputShape, this.targetShape);
    }
    return preprocessor;
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
            "Keras Flatten layer accepts only one input (received " + inputType.length + ")");
      }
    ReshapePreprocessor reshape = (ReshapePreprocessor) getInputPreprocessor(inputType);
    return reshape.getOutputType(inputType[0]);
  }
}
