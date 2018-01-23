/*
 * Copyright (c) 2018 by Andrew Charneski.
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

import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.stream.IntStream;

/**
 * The type Simple trainAll.
 */
public class SimpleEval implements Callable<SimpleEval> {
  private final Tensor[] input;
  private final NNLayer layer;
  private Tensor[] derivative;
  private Tensor output;
  
  /**
   * Instantiates a new Simple trainAll.
   *
   * @param layer the layer
   * @param input the input
   */
  public SimpleEval(final NNLayer layer, final Tensor... input) {
    this.layer = layer;
    this.input = input;
  }
  
  /**
   * Run simple trainAll.
   *
   * @param layer  the layer
   * @param tensor the tensor
   * @return the simple trainAll
   */
  public static SimpleEval run(final NNLayer layer, final Tensor... tensor) {
    return new SimpleEval(layer, tensor).call();
  }
  
  @Override
  public SimpleEval call() {
    derivative = Arrays.stream(input).map(input -> new Tensor(input.getDimensions())).toArray(i -> new Tensor[i]);
    final NNResult[] inputR = IntStream.range(0, input.length).mapToObj(i -> {
      return new NNResult((final DeltaSet<NNLayer> buffer, final TensorList data) -> {
        data.stream().forEach(t -> derivative[i].addInPlace(t));
      }, input[i]) {
        
        @Override
        public boolean isAlive() {
          return true;
        }
    
      };
    }).toArray(i -> new NNResult[i]);
    final NNResult eval = layer.eval(inputR);
    TensorList tensorList = getFeedback(eval.getData());
    eval.accumulate(new DeltaSet<NNLayer>(), tensorList);
    tensorList.freeRef();
    output = eval.getData().get(0);
    return this;
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
   * Gets feedback.
   *
   * @param data the data
   * @return the feedback
   */
  public TensorList getFeedback(final TensorList data) {
    return new TensorArray(data.stream().map(t -> t.map(v -> 1.0)).toArray(i -> new Tensor[i]));
  }
  
  /**
   * Gets output.
   *
   * @return the output
   */
  public Tensor getOutput() {
    return output;
  }
}
