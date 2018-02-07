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
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.stream.IntStream;

/**
 * The type Simple list trainAll.
 */
public class SimpleListEval extends ReferenceCountingBase implements Callable<SimpleResult>, SimpleResult {
  private final @NotNull TensorList[] input;
  private final @NotNull NNLayer layer;
  private TensorList[] derivative;
  private TensorList output;
  
  /**
   * Instantiates a new Simple list trainAll.
   *
   * @param layer the layer
   * @param input the input
   */
  public SimpleListEval(final @NotNull NNLayer layer, final @NotNull TensorList... input) {
    this.layer = layer;
    this.input = input;
    for (@NotNull TensorList x : input) x.addRef();
    layer.addRef();
  }
  
  /**
   * Accumulate.
   *
   * @param buffer the buffer
   * @param data   the data
   */
  public static void accumulate(final @NotNull TensorList buffer, final @NotNull TensorList data) {
    IntStream.range(0, data.length()).forEach(b -> {
      Tensor r = data.get(b);
      Tensor l = buffer.get(b);
      l.addInPlace(r);
      l.freeRef();
      r.freeRef();
    });
  }
  
  /**
   * Run simple list trainAll.
   *
   * @param layer  the layer
   * @param tensor the tensor
   * @return the simple list trainAll
   */
  public static @NotNull SimpleResult run(final @NotNull NNLayer layer, final TensorList... tensor) {
    return new SimpleListEval(layer, tensor).call();
  }
  
  @Override
  protected void _free() {
    for (@NotNull TensorList x : input) x.freeRef();
    layer.freeRef();
    for (@NotNull TensorList x : derivative) x.freeRef();
    output.freeRef();
  }
  
  @Override
  public @NotNull SimpleResult call() {
    TensorList[] inputCopy = Arrays.stream(input).map(x -> x.copy()).toArray(i -> new TensorList[i]);
    derivative = Arrays.stream(inputCopy).map(tensorList -> TensorArray.wrap(tensorList.stream()
                                                                                       .map(i -> new Tensor(i.getDimensions()))
                                                                                       .toArray(i -> new Tensor[i]))
                                             ).toArray(i -> new TensorList[i]);
    NNResult[] inputs = IntStream.range(0, inputCopy.length).mapToObj(i -> {
      return new NNResult(inputCopy[i], (final @NotNull DeltaSet<NNLayer> buffer, final @NotNull TensorList data) -> {
        SimpleListEval.accumulate(derivative[i], data);
      }) {
        @Override
        public boolean isAlive() {
          return true;
        }
      };
    }).toArray(i -> new NNResult[i]);
    final NNResult eval = layer.eval(inputs);
    for (@NotNull NNResult nnResult : inputs) {
      nnResult.freeRef();
    }
    TensorList outputData = eval.getData().copy();
    for (@NotNull TensorList tensorList : inputCopy) {
      tensorList.freeRef();
    }
    eval.getData().freeRef();
    @NotNull TensorList tensorList = getFeedback(outputData);
    @NotNull DeltaSet<NNLayer> buffer = new DeltaSet<>();
    eval.accumulate(buffer, tensorList);
    buffer.freeRef();
    eval.freeRef();
    tensorList.freeRef();
    output = outputData;
    return this;
  }
  
  /**
   * Get derivative tensor list [ ].
   *
   * @return the tensor list [ ]
   */
  @Override
  public TensorList[] getDerivative() {
    return derivative;
  }
  
  /**
   * Gets feedback.
   *
   * @param data the data
   * @return the feedback
   */
  public @NotNull TensorList getFeedback(final @NotNull TensorList data) {
    return TensorArray.wrap(data.stream().map(t -> t.map(v -> 1.0)).toArray(i -> new Tensor[i]));
  }
  
  /**
   * Gets output.
   *
   * @return the output
   */
  @Override
  public TensorList getOutput() {
    return output;
  }
  
  /**
   * Gets output and free.
   *
   * @return the output and free
   */
  public TensorList getOutputAndFree() {
    output.addRef();
    freeRef();
    return output;
  }
}
