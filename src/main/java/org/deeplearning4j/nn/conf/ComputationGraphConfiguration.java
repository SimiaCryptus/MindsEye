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

package org.deeplearning4j.nn.conf;

import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;

import java.util.Map;

public class ComputationGraphConfiguration {
  public class GraphBuilder {
    private InputType[] inputTypes;
    private String[] outputs;
    private Map<String, InputPreProcessor> inputPreProcessors;
    
    public void addInputs(String[] inputLayerNameArray) {
    }
    
    public void setInputTypes(InputType[] inputTypes) {
      this.inputTypes = inputTypes;
    }
    
    public void setOutputs(String[] outputs) {
      this.outputs = outputs;
    }
    
    public void addLayer(String layerName, Layer layer, String[] inboundLayerNamesArray) {
      throw new RuntimeException("NI");
    }
    
    public void addVertex(String layerName, GraphVertex vertex, String[] inboundLayerNamesArray) {
      throw new RuntimeException("NI");
    }
    
    public void setInputPreProcessors(Map<String, InputPreProcessor> inputPreProcessors) {
      this.inputPreProcessors = inputPreProcessors;
    }
    
    public NeuralNetConfiguration.ListBuilder backpropType(BackpropType truncatedBPTT) {
      throw new RuntimeException("NI");
    }
    
    public ComputationGraphConfiguration build() {
      throw new RuntimeException("NI");
    }
  }
}
