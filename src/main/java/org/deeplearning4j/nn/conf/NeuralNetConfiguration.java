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

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;

public class NeuralNetConfiguration {
  private KerasLayer layer;
  
  public KerasLayer getLayer() {
    return layer;
  }
  
  public static class Builder {
    public ListBuilder list() {
      return null;
    }
  
    public ComputationGraphConfiguration.GraphBuilder graphBuilder() {
      throw new RuntimeException("NI");
    }
  }
  
  public class ListBuilder {
    private InputType inputType;
    
    public ListBuilder inputPreProcessor(int layerIndex, InputPreProcessor preprocessor) {
      throw new RuntimeException("NI");
    }
    
    public ListBuilder layer(int i, Layer layer) {
      throw new RuntimeException("NI");
    }
    
    public ListBuilder setInputType(InputType inputType) {
      this.inputType = inputType;
      return this;
    }
    
    public ListBuilder backpropType(BackpropType standard) {
      throw new RuntimeException("NI");
    }
    
    public ListBuilder tBPTTForwardLength(int truncatedBPTT) {
      throw new RuntimeException("NI");
    }
    
    public ListBuilder tBPTTBackwardLength(int truncatedBPTT) {
      throw new RuntimeException("NI");
    }
    
    public MultiLayerConfiguration build() {
      throw new RuntimeException("NI");
    }
  }
}
