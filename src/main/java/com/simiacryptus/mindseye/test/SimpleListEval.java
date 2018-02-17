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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.stream.IntStream;

/**
 * The type Simple list trainAll.
 */
public class SimpleListEval extends ReferenceCountingBase implements Callable<SimpleResult>, SimpleResult {
  @javax.annotation.Nonnull
  private final TensorList[] input;
  @javax.annotation.Nonnull
  private final Layer layer;
  private TensorList[] derivative;
  private TensorList output;
  
  /**
   * Instantiates a new Simple list trainAll.
   *
   * @param layer the layer
   * @param input the input
   */
  public SimpleListEval(@javax.annotation.Nonnull final Layer layer, @javax.annotation.Nonnull final TensorList... input) {
    this.layer = layer;
    this.input = input;
    for (@javax.annotation.Nonnull TensorList x : input) x.addRef();
    layer.addRef();
  }
  
  /**
   * Accumulate.
   *
   * @param buffer the buffer
   * @param data   the data
   */
  public static void accumulate(@javax.annotation.Nonnull final TensorList buffer, @javax.annotation.Nonnull final TensorList data) {
    IntStream.range(0, data.length()).forEach(b -> {
      @Nullable Tensor r = data.get(b);
      @Nullable Tensor l = buffer.get(b);
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
  @javax.annotation.Nonnull
  public static SimpleResult run(@javax.annotation.Nonnull final Layer layer, final TensorList... tensor) {
    return new SimpleListEval(layer, tensor).call();
  }
  
  @Override
  protected void _free() {
    if (null != input) for (@javax.annotation.Nonnull TensorList x : input) x.freeRef();
    layer.freeRef();
    if (null != derivative) for (@javax.annotation.Nonnull TensorList x : derivative) x.freeRef();
    if (null != output) output.freeRef();
  }
  
  @javax.annotation.Nonnull
  @Override
  public SimpleResult call() {
    TensorList[] inputCopy = Arrays.stream(input).map(x -> x.copy()).toArray(i -> new TensorList[i]);
    derivative = Arrays.stream(inputCopy).map(tensorList -> TensorArray.wrap(tensorList.stream()
      .map(i -> {
        @Nonnull Tensor tensor = new Tensor(i.getDimensions());
        i.freeRef();
        return tensor;
      })
      .toArray(i -> new Tensor[i]))
    ).toArray(i -> new TensorList[i]);
    NNResult[] inputs = IntStream.range(0, inputCopy.length).mapToObj(i -> {
      return new NNResult(inputCopy[i], (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
        SimpleListEval.accumulate(derivative[i], data);
      }) {
        @Override
        public boolean isAlive() {
          return true;
        }
      };
    }).toArray(i -> new NNResult[i]);
    @Nullable final NNResult eval = layer.eval(inputs);
    for (@javax.annotation.Nonnull NNResult nnResult : inputs) {
      nnResult.freeRef();
    }
    TensorList outputData = eval.getData().copy();
    for (@javax.annotation.Nonnull TensorList tensorList : inputCopy) {
      tensorList.freeRef();
    }
    eval.getData().freeRef();
    @javax.annotation.Nonnull TensorList tensorList = getFeedback(outputData);
    @javax.annotation.Nonnull DeltaSet<Layer> buffer = new DeltaSet<>();
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
  @javax.annotation.Nonnull
  public TensorList getFeedback(@javax.annotation.Nonnull final TensorList data) {
    return TensorArray.wrap(data.stream().map(t -> {
      @Nullable Tensor map = t.map(v -> 1.0);
      t.freeRef();
      return map;
    }).toArray(i -> new Tensor[i]));
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
