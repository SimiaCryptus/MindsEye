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
public class SimpleEval extends ReferenceCountingBase implements Callable<SimpleEval> {
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
    for (Tensor x : input) x.addRef();
    layer.addRef();
  }
  
  @Override
  protected void _free() {
    for (Tensor x : input) x.freeRef();
    layer.freeRef();
    for (Tensor x : derivative) x.freeRef();
    output.freeRef();
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
    Tensor[] inputCopy = Arrays.stream(input).map(x -> x.copy()).toArray(i -> new Tensor[i]);
    derivative = Arrays.stream(inputCopy).map(input -> new Tensor(input.getDimensions())).toArray(i -> new Tensor[i]);
    NNResult[] input = IntStream.range(0, inputCopy.length).mapToObj(i -> {
      return new NNResult(TensorArray.create(inputCopy[i]), (final DeltaSet<NNLayer> buffer, final TensorList data) -> {
        data.stream().forEach(t -> derivative[i].addInPlace(t));
      }) {
        @Override
        public boolean isAlive() {
          return true;
        }
      };
    }).toArray(i -> new NNResult[i]);
    final NNResult eval = layer.eval(input);
    for (NNResult nnResult : input) {
      nnResult.getData().freeRef();
      nnResult.freeRef();
    }
    TensorList outputData = eval.getData().copy();
    Tensor tensor1 = outputData.get(0);
    output = tensor1.copy();
    tensor1.freeRef();
    for (Tensor tensor : inputCopy) {
      tensor.freeRef();
    }
    eval.getData().freeRef();
    TensorList tensorList = getFeedback(outputData);
    outputData.freeRef();
    DeltaSet<NNLayer> deltaSet = new DeltaSet<>();
    eval.accumulate(deltaSet, tensorList);
    eval.freeRef();
    deltaSet.freeRef();
    tensorList.freeRef();
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
    return TensorArray.wrap(data.stream().map(t -> t.map(v -> 1.0)).toArray(i -> new Tensor[i]));
  }
  
  /**
   * Gets output.
   *
   * @return the output
   */
  public Tensor getOutput() {
    return output;
  }
  
  public Tensor getOutputAndFree() {
    output.addRef();
    freeRef();
    return output;
  }
}
