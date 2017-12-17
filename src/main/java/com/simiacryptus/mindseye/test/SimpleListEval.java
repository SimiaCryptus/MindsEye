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
 * The type Simple list eval.
 */
public class SimpleListEval implements Callable<SimpleListEval> {
  private final TensorList[] input;
  private final NNLayer layer;
  private TensorList[] derivative;
  private TensorList output;
  
  /**
   * Instantiates a new Simple list eval.
   *
   * @param layer the layer
   * @param input the input
   */
  public SimpleListEval(final NNLayer layer, final TensorList... input) {
    this.layer = layer;
    this.input = input;
  }
  
  /**
   * Accumulate.
   *
   * @param buffer the buffer
   * @param data   the data
   */
  public static void accumulate(final TensorList buffer, final TensorList data) {
    IntStream.range(0, data.length()).forEach(b -> {
      buffer.get(b).accumulate(data.get(b));
    });
  }
  
  /**
   * Run simple list eval.
   *
   * @param layer  the layer
   * @param tensor the tensor
   * @return the simple list eval
   */
  public static SimpleListEval run(final NNLayer layer, final TensorList... tensor) {
    return new SimpleListEval(layer, tensor).call();
  }
  
  @Override
  public SimpleListEval call() {
    derivative = Arrays.stream(input).map(x -> new TensorArray(x.stream()
      .map(i -> new Tensor(i.getDimensions()))
      .toArray(i -> new Tensor[i]))
    ).toArray(i -> new TensorList[i]);
    final NNResult[] inputR = IntStream.range(0, input.length).mapToObj(i -> {
      return new NNResult(input[i]) {
        @Override
        public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
          SimpleListEval.accumulate(derivative[i], data);
        }
        
        @Override
        public boolean isAlive() {
          return true;
        }
      };
    }).toArray(i -> new NNResult[i]);
    output = GpuController.call(cudaExeCtx -> {
      final NNResult eval = layer.eval(cudaExeCtx, inputR);
      eval.accumulate(new DeltaSet<NNLayer>(), getFeedback(eval.getData()));
      return eval;
    }).getData();
    return this;
  }
  
  /**
   * Get derivative tensor list [ ].
   *
   * @return the tensor list [ ]
   */
  public TensorList[] getDerivative() {
    return derivative;
  }
  
  /**
   * Gets feedback.
   *
   * @param data the data
   * @return the feedback
   */
  public TensorArray getFeedback(final TensorList data) {
    return new TensorArray(data.stream().map(t -> t.map(v -> 1.0)).toArray(i -> new Tensor[i]));
  }
  
  /**
   * Gets output.
   *
   * @return the output
   */
  public TensorList getOutput() {
    return output;
  }
}
