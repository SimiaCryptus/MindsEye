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

import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;

import java.util.Map;

/**
 * Imports a Keras Merge layer as a DL4J Merge (graph) vertex.
 * <p>
 * TODO: handle axes arguments that alter merge behavior (requires changes to DL4J?)
 *
 * @author dave@skymind.io
 */
public class KerasMerge extends KerasLayer {
  
  private final String LAYER_FIELD_MODE = "mode";
  private final String LAYER_MERGE_MODE_SUM = "sum";
  private final String LAYER_MERGE_MODE_MUL = "mul";
  private final String LAYER_MERGE_MODE_CONCAT = "concat";
  private final String LAYER_MERGE_MODE_AVE = "ave";
  private final String LAYER_MERGE_MODE_COS = "cos";
  private final String LAYER_MERGE_MODE_DOT = "dot";
  private final String LAYER_MERGE_MODE_MAX = "max";
  
  private ElementWiseVertex.Op mergeMode = null;
  
  /**
   * Pass-through constructor from KerasLayer
   *
   * @param kerasVersion major keras version
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasMerge(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
    super(kerasVersion);
  }
  
  /**
   * Constructor from parsed Keras layer configuration dictionary.
   *
   * @param layerConfig dictionary containing Keras layer configuration.
   * @throws InvalidKerasConfigurationException
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasMerge(Map<String, Object> layerConfig)
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
  public KerasMerge(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    super(layerConfig, enforceTrainingConfig);
    this.mergeMode = getMergeMode(layerConfig);
    if (this.mergeMode == null) {
      this.vertex = new MergeVertex();
    }
    else {
      this.vertex = new ElementWiseVertex(mergeMode);
    }
  }
  
  public ElementWiseVertex.Op getMergeMode(Map<String, Object> layerConfig)
    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
    if (!innerConfig.containsKey(LAYER_FIELD_MODE)) {
      throw new InvalidKerasConfigurationException(
        "Keras Merge layer config missing " + LAYER_FIELD_MODE + " field");
    }
    ElementWiseVertex.Op op = null;
    String mergeMode = (String) innerConfig.get(LAYER_FIELD_MODE);
    switch (mergeMode) {
      case LAYER_MERGE_MODE_SUM:
        op = ElementWiseVertex.Op.Add;
        break;
      case LAYER_MERGE_MODE_MUL:
        op = ElementWiseVertex.Op.Product;
        break;
      case LAYER_MERGE_MODE_CONCAT:
        // leave null
        break;
      case LAYER_MERGE_MODE_AVE:
        throw new RuntimeException("NI");
//                op = ElementWiseVertex.Op.Average;
//                break;
      case LAYER_MERGE_MODE_MAX:
        throw new RuntimeException("NI");
//                op = ElementWiseVertex.Op.Max;
//                break;
      case LAYER_MERGE_MODE_COS:
      case LAYER_MERGE_MODE_DOT:
      default:
        throw new UnsupportedKerasConfigurationException(
          "Keras Merge layer mode " + mergeMode + " not supported");
    }
    return op;
  }
  
  /**
   * Get layer output type.
   *
   * @param inputType Array of InputTypes
   * @return output type as InputType
   * @throws InvalidKerasConfigurationException
   */
  @Override
  public InputType getOutputType(InputType... inputType) {
    return this.vertex.getOutputType(-1, inputType);
  }
}
