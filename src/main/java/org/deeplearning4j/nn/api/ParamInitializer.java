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

package org.deeplearning4j.nn.api;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;

public interface ParamInitializer {
  int numParams(NeuralNetConfiguration conf);
  
  int numParams(Layer layer);
  
  List<String> paramKeys(Layer layer);
  
  List<String> weightKeys(Layer layer);
  
  List<String> biasKeys(Layer layer);
  
  boolean isWeightParam(Layer layer, String key);
  
  boolean isBiasParam(Layer layer, String key);
  
  Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams);
  
  Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView);
}
