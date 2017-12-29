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


import com.simiacryptus.mindseye.opt.IterativeTrainer;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.Hdf5Archive;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.config.KerasModelConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.StringUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.core.type.TypeReference;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Utility functionality to import keras models
 *
 * @author Max Pumperla
 */
public class KerasModelUtils {
  private static final Logger log = LoggerFactory.getLogger(IterativeTrainer.class);
  
  /**
   * Helper function to import weights from nested Map into existing model. Depends critically
   * on matched layer and parameter names. In general this seems to be straightforward for most
   * Keras models and layersOrdered, but there may be edge cases.
   *
   * @param model DL4J Model interface
   * @return DL4J Model interface
   * @throws InvalidKerasConfigurationException
   */
  public static Model copyWeightsToModel(Model model, Map<String, KerasLayer> layers)
    throws InvalidKerasConfigurationException {
        /* Get list if layers from model. */
    Layer[] layersFromModel;
      if (model instanceof MultiLayerNetwork) {
          layersFromModel = ((MultiLayerNetwork) model).getLayers();
      }
      else {
          layersFromModel = ((ComputationGraph) model).getLayers();
      }

        /* Iterate over layers in model, setting weights when relevant. */
    Set<String> layerNames = new HashSet<>(layers.keySet());
    for (Layer layer : layersFromModel) {
      String layerName = layer.conf().getLayer().getLayerName();
        if (!layers.containsKey(layerName)) {
            throw new InvalidKerasConfigurationException(
              "No weights found for layer in model (named " + layerName + ")");
        }
      layers.get(layerName).copyWeightsToLayer(layer);
      layerNames.remove(layerName);
    }
    
    for (String layerName : layerNames) {
        if (layers.get(layerName).getNumParams() > 0) {
            throw new InvalidKerasConfigurationException(
              "Attemping to copy weights for layer not in model (named " + layerName + ")");
        }
    }
    return model;
  }
  
  /**
   * Determine Keras major version
   *
   * @param modelConfig parsed model configuration for keras model
   * @param config      basic model configuration (KerasModelConfiguration)
   * @return Major Keras version (1 or 2)
   * @throws InvalidKerasConfigurationException
   */
  public static int determineKerasMajorVersion(Map<String, Object> modelConfig, KerasModelConfiguration config)
    throws InvalidKerasConfigurationException {
    int kerasMajorVersion;
    if (!modelConfig.containsKey(config.getFieldKerasVersion())) {
      log.warn("Could not read keras version used (no "
        + config.getFieldKerasVersion() + " field found) \n"
        + "assuming keras version is 1.0.7 or earlier."
      );
      kerasMajorVersion = 1;
    }
    else {
      String kerasVersionString = (String) modelConfig.get(config.getFieldKerasVersion());
      if (Character.isDigit(kerasVersionString.charAt(0))) {
        kerasMajorVersion = Character.getNumericValue(kerasVersionString.charAt(0));
      }
      else {
        throw new InvalidKerasConfigurationException(
          "Keras version was not readable (" + config.getFieldKerasVersion() + " provided)"
        );
      }
    }
    return kerasMajorVersion;
  }
  
  /**
   * Determine Keras backend
   *
   * @param modelConfig parsed model configuration for keras model
   * @param config      basic model configuration (KerasModelConfiguration)
   * @return Keras backend string
   * @throws InvalidKerasConfigurationException
   */
  public static String determineKerasBackend(Map<String, Object> modelConfig, KerasModelConfiguration config)
    throws InvalidKerasConfigurationException {
    String kerasBackend = null;
    if (!modelConfig.containsKey(config.getFieldBackend())) {
      log.warn("Could not read keras backend used (no "
        + config.getFieldBackend() + " field found) \n"
      );
    }
    else {
      kerasBackend = (String) modelConfig.get(config.getFieldBackend());
    }
    return kerasBackend;
  }
  
  public static String findParameterName(String parameter, String[] fragmentList) {
    Matcher layerNameMatcher =
      Pattern.compile(fragmentList[fragmentList.length - 1]).matcher(parameter);
      if (!(layerNameMatcher.find())) {
          log.warn("Unable to match layer parameter name " + parameter + " for stored weights.");
      }
    String parameterNameFound = layerNameMatcher.replaceFirst("");

        /* Usually layer name is separated from parameter name by an underscore. */
    Matcher paramNameMatcher = Pattern.compile("^_(.+)$").matcher(parameterNameFound);
      if (paramNameMatcher.find()) {
          parameterNameFound = paramNameMatcher.group(1);
      }

        /* TensorFlow backend often appends ":" followed by one or more digits to parameter names. */
    Matcher tfSuffixMatcher = Pattern.compile(":\\d+?$").matcher(parameterNameFound);
      if (tfSuffixMatcher.find()) {
          parameterNameFound = tfSuffixMatcher.replaceFirst("");
      }

        /* TensorFlow backend also may append "_" followed by one or more digits to parameter names.*/
    Matcher tfParamNbMatcher = Pattern.compile("_\\d+$").matcher(parameterNameFound);
      if (tfParamNbMatcher.find()) {
          parameterNameFound = tfParamNbMatcher.replaceFirst("");
      }
    
    return parameterNameFound;
  }
  
  /**
   * Store weights to import with each associated Keras layer.
   *
   * @param weightsArchive Hdf5Archive
   * @param weightsRoot
   * @throws InvalidKerasConfigurationException
   */
  public static void importWeights(Hdf5Archive weightsArchive, String weightsRoot, Map<String, KerasLayer> layers)
    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    // check to ensure naming scheme doesn't include forward slash
    boolean includesSlash = false;
    for (String layerName : layers.keySet()) {
        if (layerName.contains("/")) {
            includesSlash = true;
        }
    }
    List<String> layerGroups;
    if (!includesSlash) {
      layerGroups = weightsRoot != null ? weightsArchive.getGroups(weightsRoot) : weightsArchive.getGroups();
    }
    else {
      layerGroups = new ArrayList<>(layers.keySet());
    }
        /* Set weights in KerasLayer for each entry in weights map. */
    for (String layerName : layerGroups) {
      List<String> layerParamNames;
      
      // there's a bug where if a layer name contains a forward slash, the first fragment must be appended
      // to the name of the dataset...it appears h5 interprets the forward slash as a data group
      String[] layerFragments = layerName.split("/");
      
      // Find nested groups when using Tensorflow
      String rootPrefix = weightsRoot != null ? weightsRoot + "/" : "";
      List<String> attributeStrParts = new ArrayList<>();
      String attributeStr = weightsArchive.readAttributeAsString(
        "weight_names", rootPrefix + layerName
      );
      String attributeJoinStr;
      Matcher attributeMatcher = Pattern.compile(":\\d+").matcher(attributeStr);
      Boolean foundTfGroups = attributeMatcher.find();
      
      if (foundTfGroups) {
        for (String part : attributeStr.split("/")) {
          part = part.trim();
            if (part.length() == 0) {
                break;
            }
          Matcher tfSuffixMatcher = Pattern.compile(":\\d+").matcher(part);
            if (tfSuffixMatcher.find()) {
                break;
            }
          attributeStrParts.add(part);
        }
        attributeJoinStr = StringUtils.join("/", attributeStrParts);
      }
      else {
        attributeJoinStr = layerFragments[0];
      }
      
      String baseAttributes = layerName + "/" + attributeJoinStr;
      if (layerFragments.length > 1) {
        try {
          layerParamNames = weightsArchive.getDataSets(rootPrefix + baseAttributes);
        } catch (Exception e) {
          layerParamNames = weightsArchive.getDataSets(rootPrefix + layerName);
        }
      }
      else {
        if (foundTfGroups) {
          layerParamNames = weightsArchive.getDataSets(rootPrefix + baseAttributes);
        }
        else {
          layerParamNames = weightsArchive.getDataSets(rootPrefix + layerName);
          
        }
      }
        if (layerParamNames.isEmpty()) {
            continue;
        }
        if (!layers.containsKey(layerName)) {
            throw new InvalidKerasConfigurationException(
              "Found weights for layer not in model (named " + layerName + ")");
        }
      KerasLayer layer = layers.get(layerName);
        if (layerParamNames.size() != layer.getNumParams()) {
            throw new InvalidKerasConfigurationException(
              "Found " + layerParamNames.size() + " weights for layer with " + layer.getNumParams()
                + " trainable params (named " + layerName + ")");
        }
      Map<String, INDArray> weights = new HashMap<String, INDArray>();
      
      for (String layerParamName : layerParamNames) {
        String paramName = KerasModelUtils.findParameterName(layerParamName, layerFragments);
        INDArray paramValue;
        if (foundTfGroups) {
          paramValue = weightsArchive.readDataSet(layerParamName, rootPrefix + baseAttributes);
        }
        else {
          if (layerFragments.length > 1) {
            paramValue = weightsArchive.readDataSet(
              layerFragments[0] + "/" + layerParamName, rootPrefix, layerName);
          }
          else {
            paramValue = weightsArchive.readDataSet(layerParamName, rootPrefix, layerName);
          }
        }
        weights.put(paramName, paramValue);
      }
      layer.setWeights(weights);
    }

        /* Look for layers in model with no corresponding entries in weights map. */
    Set<String> layerNames = new HashSet<>(layers.keySet());
    layerNames.removeAll(layerGroups);
    for (String layerName : layerNames) {
        if (layers.get(layerName).getNumParams() > 0) {
            throw new InvalidKerasConfigurationException("Could not find weights required for layer " + layerName);
        }
    }
  }
  
  /**
   * Parse Keras model configuration from JSON or YAML string representation
   *
   * @param modelJson JSON string representing model (potentially null)
   * @param modelYaml YAML string representing model (potentially null)
   * @return Model configuration as Map<String, Object>
   * @throws IOException
   * @throws InvalidKerasConfigurationException
   */
  public static Map<String, Object> parseModelConfig(String modelJson, String modelYaml) throws IOException, InvalidKerasConfigurationException {
    Map<String, Object> modelConfig;
      if (modelJson != null) {
          modelConfig = parseJsonString(modelJson);
      }
      else if (modelYaml != null) {
          modelConfig = parseYamlString(modelYaml);
      }
      else {
          throw new InvalidKerasConfigurationException("Requires model configuration as either JSON or YAML string.");
      }
    return modelConfig;
  }
  
  
  /**
   * Convenience function for parsing JSON strings.
   *
   * @param json String containing valid JSON
   * @return Nested (key,value) map of arbitrary depth
   * @throws IOException
   */
  public static Map<String, Object> parseJsonString(String json) throws IOException {
    ObjectMapper mapper = new ObjectMapper();
    TypeReference<HashMap<String, Object>> typeRef = new TypeReference<HashMap<String, Object>>() {
    };
    return mapper.readValue(json, typeRef);
  }
  
  /**
   * Convenience function for parsing YAML strings.
   *
   * @param yaml String containing valid YAML
   * @return Nested (key,value) map of arbitrary depth
   * @throws IOException
   */
  public static Map<String, Object> parseYamlString(String yaml) throws IOException {
    ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
    TypeReference<HashMap<String, Object>> typeRef = new TypeReference<HashMap<String, Object>>() {
    };
    return mapper.readValue(yaml, typeRef);
  }
  
}
