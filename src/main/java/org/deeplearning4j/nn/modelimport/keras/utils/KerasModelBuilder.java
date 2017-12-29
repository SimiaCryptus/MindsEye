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

import org.apache.commons.io.IOUtils;
import org.deeplearning4j.nn.modelimport.keras.Hdf5Archive;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasSequentialModel;
import org.deeplearning4j.nn.modelimport.keras.config.KerasModelConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;

public class KerasModelBuilder implements Cloneable {
  protected String modelJson = null;
  protected String modelYaml = null;
  protected String trainingJson = null;
  protected Hdf5Archive weightsArchive = null;
  protected String weightsRoot = null;
  protected Hdf5Archive trainingArchive = null;
  protected boolean enforceTrainingConfig = false;
  protected KerasModelConfiguration config;
  
  
  public KerasModelBuilder(KerasModelConfiguration config) {
    this.config = config;
  }
  
  public KerasModelBuilder modelJson(String modelJson) {
    this.modelJson = modelJson;
    return this;
  }
  
  public KerasModelBuilder modelJsonFilename(String modelJsonFilename) throws IOException {
    this.modelJson = new String(Files.readAllBytes(Paths.get(modelJsonFilename)));
    return this;
  }
  
  public KerasModelBuilder modelJsonInputStream(InputStream modelJsonInputStream) throws IOException {
    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
    IOUtils.copy(modelJsonInputStream, byteArrayOutputStream);
    this.modelJson = new String(byteArrayOutputStream.toByteArray());
    return this;
  }
  
  public KerasModelBuilder modelYaml(String modelYaml) {
    this.modelYaml = modelYaml;
    return this;
  }
  
  public KerasModelBuilder modelYamlFilename(String modelYamlFilename) throws IOException {
    this.modelJson = new String(Files.readAllBytes(Paths.get(modelYamlFilename)));
    return this;
  }
  
  public KerasModelBuilder modelYamlInputStream(InputStream modelYamlInputStream) throws IOException {
    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
    IOUtils.copy(modelYamlInputStream, byteArrayOutputStream);
    this.modelJson = new String(byteArrayOutputStream.toByteArray());
    return this;
  }
  
  public KerasModelBuilder trainingJson(String trainingJson) {
    this.trainingJson = trainingJson;
    return this;
  }
  
  public KerasModelBuilder trainingJsonInputStream(InputStream trainingJsonInputStream) throws IOException {
    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
    IOUtils.copy(trainingJsonInputStream, byteArrayOutputStream);
    this.trainingJson = new String(byteArrayOutputStream.toByteArray());
    return this;
  }
  
  public KerasModelBuilder modelHdf5Filename(String modelHdf5Filename)
    throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException, IOException {
    this.weightsArchive = this.trainingArchive = new Hdf5Archive(modelHdf5Filename);
    this.weightsRoot = config.getTrainingWeightsRoot();
      if (!this.weightsArchive.hasAttribute(config.getTrainingModelConfigAttribute())) {
          throw new InvalidKerasConfigurationException(
            "Model configuration attribute missing from " + modelHdf5Filename + " archive.");
      }
    String initialModelJson = this.weightsArchive.readAttributeAsJson(
      config.getTrainingModelConfigAttribute());
    String kerasVersion = this.weightsArchive.readAttributeAsFixedLengthString(
      config.getFieldKerasVersion(), 5);
    Map<String, Object> modelMapper = KerasModelUtils.parseJsonString(initialModelJson);
    modelMapper.put(config.getFieldKerasVersion(), kerasVersion);
    this.modelJson = new ObjectMapper().writeValueAsString(modelMapper);
      if (this.trainingArchive.hasAttribute(config.getTrainingTrainingConfigAttribute())) {
          this.trainingJson = this.trainingArchive.readAttributeAsJson(config.getTrainingTrainingConfigAttribute());
      }
    return this;
  }
  
  public KerasModelBuilder weightsHdf5Filename(String weightsHdf5Filename) {
    this.weightsArchive = new Hdf5Archive(weightsHdf5Filename);
    return this;
  }
  
  public KerasModelBuilder enforceTrainingConfig(boolean enforceTrainingConfig) {
    this.enforceTrainingConfig = enforceTrainingConfig;
    return this;
  }
  
  
  public KerasModel buildModel()
    throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    return new KerasModel(this);
  }
  
  public KerasSequentialModel buildSequential()
    throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
    return new KerasSequentialModel(this);
  }
  
  public String getModelJson() {
    return modelJson;
  }
  
  public String getModelYaml() {
    return modelYaml;
  }
  
  public Hdf5Archive getWeightsArchive() {
    return weightsArchive;
  }
  
  public String getWeightsRoot() {
    return weightsRoot;
  }
  
  public String getTrainingJson() {
    return trainingJson;
  }
  
  public Hdf5Archive getTrainingArchive() {
    return trainingArchive;
  }
  
  public boolean isEnforceTrainingConfig() {
    return enforceTrainingConfig;
  }
}