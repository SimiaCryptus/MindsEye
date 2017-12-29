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

package org.deeplearning4j.nn.conf.layers.recurrent;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseRecurrentLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.params.SimpleRnnParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**
 * Simple RNN - aka "vanilla" RNN is the simplest type of recurrent neural network layer.
 * It implements out_t = activationFn( in_t * inWeight + out_(t-1) * recurrentWeights + bias).
 * <p>
 * Note that other architectures (LSTM, etc) are usually more effective, especially for longer time series
 *
 * @author Alex Black
 */
public class SimpleRnn extends BaseRecurrentLayer {
  
  private double l1;
  private double l1Bias;
  private double l2;
  private double l2Bias;
  private int NIn;
  private int NOut;
  
  protected SimpleRnn(Builder builder) {
    super(builder);
  }
  
  private SimpleRnn() {
    super();
    
  }
  
  @Override
  public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners,
                           int layerIndex, INDArray layerParamsView, boolean initializeParams) {
    throw new RuntimeException("NI");
//        LayerValidation.assertNInNOutSet("SimpleRnn", getLayerName(), layerIndex, getNIn(), getNOut());
//
//        org.deeplearning4j.nn.layers.recurrent.SimpleRnn ret =
//                new org.deeplearning4j.nn.layers.recurrent.SimpleRnn(conf);
//        ret.setListeners(iterationListeners);
//        ret.setIndex(layerIndex);
//        ret.setParamsViewArray(layerParamsView);
//        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
//        ret.setParamTable(paramTable);
//        ret.setConf(conf);
//        return ret;
  }
  
  @Override
  public ParamInitializer initializer() {
    return SimpleRnnParamInitializer.getInstance();
  }
  
  @Override
  public double getL1ByParam(String paramName) {
    switch (paramName) {
      case SimpleRnnParamInitializer.WEIGHT_KEY:
      case SimpleRnnParamInitializer.RECURRENT_WEIGHT_KEY:
        return l1;
      case SimpleRnnParamInitializer.BIAS_KEY:
        return l1Bias;
      default:
        throw new IllegalStateException("Unknown parameter: \"" + paramName + "\"");
    }
  }
  
  @Override
  public double getL2ByParam(String paramName) {
    switch (paramName) {
      case SimpleRnnParamInitializer.WEIGHT_KEY:
      case SimpleRnnParamInitializer.RECURRENT_WEIGHT_KEY:
        return l2;
      case SimpleRnnParamInitializer.BIAS_KEY:
        return l2Bias;
      default:
        throw new IllegalStateException("Unknown parameter: \"" + paramName + "\"");
    }
  }
  
  @Override
  public LayerMemoryReport getMemoryReport(InputType inputType) {
    return null;
  }
  
  public int getNIn() {
    return NIn;
  }
  
  public int getNOut() {
    return NOut;
  }
  
  public static class Builder extends BaseRecurrentLayer.Builder<Builder> {
    
    
    @Override
    public SimpleRnn build() {
      return new SimpleRnn(this);
    }
  }
}
