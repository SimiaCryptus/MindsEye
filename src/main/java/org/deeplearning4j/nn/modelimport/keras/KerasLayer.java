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
package org.deeplearning4j.nn.modelimport.keras;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfigurationFactory;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasRegularizerUtils;
import org.deeplearning4j.util.StringUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Build Layer from Keras layer configuration.
 *
 * @author dave@skymind.io
 */
public class KerasLayer extends Layer {
  static final Map<String, Class<? extends KerasLayer>> customLayers = new HashMap<>();
  private static final Logger log = LoggerFactory.getLogger(KerasInitilizationUtils.class);
  private static final String LAYER_FIELD_KERAS_VERSION = "keras_version";
  protected String className; // Keras layer class name
  protected String layerName; // Keras layer name
  protected int[] inputShape; // Keras layer input shape
  protected DimOrder dimOrder; // Keras layer backend dimension order
  protected List<String> inboundLayerNames; // List of inbound layers
  protected Layer layer; // Resulting DL4J layer
  protected GraphVertex vertex; // Resulting DL4J vertex
  protected Map<String, INDArray> weights; // Weights
  protected double weightL1Regularization = 0.0; // L1 regularization
  protected double weightL2Regularization = 0.0; // L2 regularization
  protected double dropout = 1.0; // Dropout
  protected Integer kerasMajorVersion = 2; // Set 2 as default for now
  protected KerasLayerConfiguration conf;
  
  /**
   * Constructor with Keras version only.
   *
   * @param kerasVersion major Keras version (1 or 2)
   * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration
   */
  protected KerasLayer(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
    this.className = null;
    this.layerName = null;
    this.inputShape = null;
    this.dimOrder = DimOrder.NONE;
    this.inboundLayerNames = new ArrayList<>();
    this.layer = null;
    this.vertex = null;
    this.weights = null;
    this.kerasMajorVersion = kerasVersion;
    this.conf = KerasLayerConfigurationFactory.get(this.kerasMajorVersion);
  }
  
  /**
   * Default constructor.
   *
   * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration
   */
  protected KerasLayer() {
    this.className = null;
    this.layerName = null;
    this.inputShape = null;
    this.dimOrder = DimOrder.NONE;
    this.inboundLayerNames = new ArrayList<>();
    this.layer = null;
    this.vertex = null;
    this.weights = null;
    try {
      this.conf = KerasLayerConfigurationFactory.get(this.kerasMajorVersion);
    } catch (UnsupportedKerasConfigurationException e) {
      throw new RuntimeException(e);
    }
    
  }
  
  /**
   * Constructor.
   *
   * @param layerConfig dictionary containing Keras layer configuration
   */
  protected KerasLayer(Map<String, Object> layerConfig)
    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    this(layerConfig, true);
  }
  
  /**
   * Constructor. "enforceTrainingConfig" parameter controls whether layer is built for
   * training. This controls behavior of certain exceptions. In training mode, passing
   * an unsupported regularizer will generate an error. In non-training mode, it
   * generates only a warning.
   *
   * @param layerConfig           dictionary containing Keras layer configuration
   * @param enforceTrainingConfig whether layer should be built for training (controls certain exceptions)
   */
  protected KerasLayer(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    this.kerasMajorVersion = (Integer) layerConfig.get(LAYER_FIELD_KERAS_VERSION);
    this.conf = KerasLayerConfigurationFactory.get(this.kerasMajorVersion);
    this.className = KerasLayerUtils.getClassNameFromConfig(layerConfig, conf);
    if (this.className == null) {
      throw new InvalidKerasConfigurationException("Keras layer class name is missing");
    }
    this.layerName = KerasLayerUtils.getLayerNameFromConfig(layerConfig, conf);
    if (this.layerName == null) {
      throw new InvalidKerasConfigurationException("Keras layer class name is missing");
    }
    this.inputShape = KerasLayerUtils.getInputShapeFromConfig(layerConfig, conf);
    this.dimOrder = KerasLayerUtils.getDimOrderFromConfig(layerConfig, conf);
    this.inboundLayerNames = KerasLayerUtils.getInboundLayerNamesFromConfig(layerConfig, conf);
    this.layer = null;
    this.vertex = null;
    this.weights = null;
    
    this.weightL1Regularization = KerasRegularizerUtils.getWeightRegularizerFromConfig(
      layerConfig, conf, conf.getLAYER_FIELD_W_REGULARIZER(), conf.getREGULARIZATION_TYPE_L1());
    this.weightL2Regularization = KerasRegularizerUtils.getWeightRegularizerFromConfig(
      layerConfig, conf, conf.getLAYER_FIELD_W_REGULARIZER(), conf.getREGULARIZATION_TYPE_L2());
    this.dropout = KerasLayerUtils.getDropoutFromConfig(layerConfig, conf);
    KerasLayerUtils.checkForUnsupportedConfigurations(layerConfig, enforceTrainingConfig, conf);
  }
  
  /**
   * Register a custom layer
   *
   * @param layerName   name of custom layer
   * @param configClass class of custom layer
   */
  public static void registerCustomLayer(String layerName, Class<? extends KerasLayer> configClass) {
    customLayers.put(layerName, configClass);
  }
  
  /**
   * Get Keras major version of this layer.
   *
   * @return Keras version as integer
   */
  public Integer getKerasMajorVersion() {
    return this.kerasMajorVersion;
  }
  
  /**
   * Get Keras layer class name.
   *
   * @return Keras layer class name
   */
  public String getClassName() {
    return this.className;
  }
  
  /**
   * Get Keras layer name.
   *
   * @return layer name
   */
  public String getLayerName() {
    return this.layerName;
  }
  
  /**
   * Get layer input shape.
   *
   * @return input shape
   */
  public int[] getInputShape() {
    if (this.inputShape == null) {
      return null;
    }
    return this.inputShape.clone();
  }
  
  /**
   * Get Keras layer backend dimension order.
   *
   * @return Keras layer (backend) dimension order
   */
  protected DimOrder getDimOrder() {
    return this.dimOrder;
  }
  
  /**
   * Set Keras layer backend dimension order.
   */
  void setDimOrder(DimOrder dimOrder) {
    this.dimOrder = dimOrder;
  }
  
  /**
   * Get list of inbound layers.
   *
   * @return list of inbound layer names
   */
  List<String> getInboundLayerNames() {
    if (this.inboundLayerNames == null) {
      this.inboundLayerNames = new ArrayList<>();
    }
    return this.inboundLayerNames;
  }
  
  /**
   * Set list of inbound layers.
   *
   * @param inboundLayerNames list of inbound layer naems
   */
  void setInboundLayerNames(List<String> inboundLayerNames) {
    this.inboundLayerNames = new ArrayList<>(inboundLayerNames);
  }
  
  /**
   * Returns number of trainable parameters in layer.
   *
   * @return number of trainable parameters
   */
  public int getNumParams() {
    return 0;
  }
  
  /**
   * Indicates whether layer uses regularization.
   *
   * @return boolean
   */
  public boolean usesRegularization() {
    return (this.weightL1Regularization > 0.0 || this.weightL2Regularization > 0.0 || this.dropout < 1.0);
  }
  
  /**
   * Set weights for Keras layer.
   *
   * @param weights Map of named NDArrays
   */
  public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
    //no op
  }
  
  /**
   * Copy Keras layer weights to DL4J Layer.
   *
   * @param layer DL4J layer
   * @throws InvalidKerasConfigurationException Invalid Keras configuration
   */
  public void copyWeightsToLayer(org.deeplearning4j.nn.api.Layer layer) throws InvalidKerasConfigurationException {
    if (this.getNumParams() > 0) {
      String dl4jLayerName = layer.conf().getLayer().getLayerName();
      String kerasLayerName = this.getLayerName();
      String msg = "Error when attempting to copy weights from Keras layer " + kerasLayerName + " to DL4J layer "
        + dl4jLayerName;
  
      if (this.weights == null) {
        throw new InvalidKerasConfigurationException(msg + "(weights is null)");
      }
      
      Set<String> paramsInLayer = new HashSet<>(layer.paramTable().keySet());
      Set<String> paramsInKerasLayer = new HashSet<>(this.weights.keySet());

            /* Check for parameters in layer for which we don't have weights. */
      paramsInLayer.removeAll(paramsInKerasLayer);
      if (!paramsInLayer.isEmpty()) {
        String joinedParamsInLayer = StringUtils.join(", ", paramsInLayer);
        throw new InvalidKerasConfigurationException(
          msg + "(no stored weights for parameters: " + joinedParamsInLayer + ")");
      }

            /* Check for parameters NOT in layer for which we DO have weights. */
      paramsInKerasLayer.removeAll(layer.paramTable().keySet());
      if (!paramsInKerasLayer.isEmpty()) {
        String joinedParamsInKerasLayer = StringUtils.join(", ", paramsInKerasLayer);
        throw new InvalidKerasConfigurationException(
          msg + "(found no parameters named: " + joinedParamsInKerasLayer + ")");
      }

            /* Copy weights. */
      for (String paramName : layer.paramTable().keySet()) {
        layer.setParam(paramName, this.weights.get(paramName));
      }
    }
  }
  
  /**
   * Whether this Keras layer maps to a DL4J Layer.
   *
   * @return true or false
   */
  public boolean isLayer() {
    return this.layer != null;
  }
  
  /**
   * Gets corresponding DL4J Layer, if any.
   *
   * @return DL4J Layer
   * @see org.deeplearning4j.nn.api.Layer
   */
  public Layer getLayer() {
    return this.layer;
  }
  
  /**
   * Whether this Keras layer maps to a DL4J Vertex.
   *
   * @return true or false
   */
  public boolean isVertex() {
    return this.vertex != null;
  }
  
  /**
   * Gets corresponding DL4J Vertex, if any.
   *
   * @return DL4J Vertex
   * @see GraphVertex
   */
  public GraphVertex getVertex() {
    return this.vertex;
  }
  
  /**
   * Whether this Keras layer maps to a DL4J InputPreProcessor.
   *
   * @return true or false
   */
  public boolean isInputPreProcessor() {
    return false;
  }
  
  /**
   * Gets appropriate DL4J InputPreProcessor for given InputTypes.
   *
   * @param inputType Array of InputTypes
   * @return DL4J InputPreProcessor
   * @throws InvalidKerasConfigurationException Invalid Keras configuration
   * @see InputPreProcessor
   */
  public InputPreProcessor getInputPreprocessor(InputType... inputType) throws InvalidKerasConfigurationException {
    InputPreProcessor preprocessor = null;
    if (this.layer != null) {
      if (inputType.length > 1) {
        throw new InvalidKerasConfigurationException(
          "Keras layer of type \"" + this.className + "\" accepts only one input");
      }
      preprocessor = this.layer.getPreProcessorForInputType(inputType[0]);
    }
    return preprocessor;
  }
  
  /**
   * Get layer output type.
   *
   * @param inputType Array of InputTypes
   * @return output type as InputType
   * @throws InvalidKerasConfigurationException Invalid Keras configuration
   */
  public InputType getOutputType(InputType... inputType)
    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    throw new UnsupportedOperationException(
      "Cannot determine output type for Keras layer of type " + this.className);
  }
  
  /**
   * Indicates whether this layer a valid inbound layer. Currently, only
   * (known) DL4J Layers and inputs are valid inbound layers. "Preprocessor"
   * layers (reshaping, merging, etc.) are replaced by their own inbound layers.
   *
   * @return boolean indicating whether layer is valid inbound layer
   * @see org.deeplearning4j.nn.api.Layer
   */
  public boolean isValidInboundLayer() throws InvalidKerasConfigurationException {
    return (getLayer() != null || getVertex() != null || getInputPreprocessor() != null
      || this.className.equals(conf.getLAYER_CLASS_NAME_INPUT()));
  }
  
  public enum DimOrder {NONE, THEANO, TENSORFLOW}
}
