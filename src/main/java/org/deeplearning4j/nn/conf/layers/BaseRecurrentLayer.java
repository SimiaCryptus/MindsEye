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

package org.deeplearning4j.nn.conf.layers;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

public abstract class BaseRecurrentLayer extends KerasLayer {
  public BaseRecurrentLayer(Builder builder) {
    super();
    
  }
  
  public BaseRecurrentLayer() {
    super();
    
  }
  
  public abstract org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners,
                                                              int layerIndex, INDArray layerParamsView, boolean initializeParams);
  
  public abstract ParamInitializer initializer();
  
  public abstract double getL1ByParam(String paramName);
  
  public abstract double getL2ByParam(String paramName);
  
  public abstract LayerMemoryReport getMemoryReport(InputType inputType);
  
  public abstract static class Builder<T> {
    public abstract SimpleRnn build();
  }
}
