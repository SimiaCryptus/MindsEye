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

package com.simiacryptus.mindseye.test;

import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;

import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.stream.IntStream;

/**
 * The type Simple eval.
 */
public class SimpleEval implements Callable<SimpleEval> {
  private final NNLayer layer;
  private final Tensor[] input;
  private Tensor[] derivative;
  private Tensor output;
  
  /**
   * Instantiates a new Simple eval.
   *
   * @param layer the layer
   * @param input the input
   */
  public SimpleEval(NNLayer layer, Tensor... input) {
    this.layer = layer;
    this.input = input;
  }
  
  /**
   * Run simple eval.
   *
   * @param layer  the layer
   * @param tensor the tensor
   * @return the simple eval
   */
  public static SimpleEval run(NNLayer layer, Tensor... tensor) {
    return new SimpleEval(layer, tensor).call();
  }
  
  /**
   * Get derivative tensor [ ].
   *
   * @return the tensor [ ]
   */
  public Tensor[] getDerivative() {
    return derivative;
  }
  
  /**
   * Gets output.
   *
   * @return the output
   */
  public Tensor getOutput() {
    return output;
  }
  
  @Override
  public SimpleEval call() {
    derivative = Arrays.stream(input).map(input -> new Tensor(input.getDimensions())).toArray(i -> new Tensor[i]);
    NNResult[] inputR = IntStream.range(0, input.length).mapToObj(i -> {
      return new NNResult(input[i]) {
        @Override
        public void accumulate(DeltaSet buffer, TensorList data) {
          data.stream().forEach(t -> derivative[i].accum(t));
        }
        
        @Override
        public boolean isAlive() {
          return true;
        }
      };
    }).toArray(i -> new NNResult[i]);
    NNResult result = GpuController.call(cudaExeCtx -> {
      NNResult eval = layer.eval(cudaExeCtx, inputR);
      eval.accumulate(new DeltaSet(), getFeedback(eval.getData()));
      return eval;
    });
    output = result.getData().get(0);
    return this;
  }
  
  /**
   * Gets feedback.
   *
   * @param data the data
   * @return the feedback
   */
  public TensorArray getFeedback(TensorList data) {
    return new TensorArray(data.stream().map(t -> t.map(v -> 1.0)).toArray(i -> new Tensor[i]));
  }
}
