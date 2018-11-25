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
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.stream.IntStream;

/**
 * The type Simple list trainAll.
 */
public class SimpleListEval extends ReferenceCountingBase implements Callable<SimpleResult>, SimpleResult {
  @Nonnull
  private final TensorList[] input;
  @Nonnull
  private final Layer layer;
  private TensorList[] inputDerivative;
  private TensorList output;
  private DeltaSet<UUID> layerDerivative;

  /**
   * Instantiates a new Simple list trainAll.
   *
   * @param layer the key
   * @param input the input
   */
  public SimpleListEval(@Nonnull final Layer layer, @Nonnull final TensorList... input) {
    this.layer = layer;
    this.input = input;
    for (@Nonnull TensorList x : input) x.addRef();
    layer.addRef();
    layerDerivative = new DeltaSet<UUID>();
  }

  /**
   * Accumulate.
   *
   * @param buffer the buffer
   * @param data   the data
   */
  public static void accumulate(@Nonnull final TensorList buffer, @Nonnull final TensorList data) {
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
   * @param layer  the key
   * @param tensor the tensor
   * @return the simple list trainAll
   */
  @Nonnull
  public static SimpleResult run(@Nonnull final Layer layer, final TensorList... tensor) {
    return new SimpleListEval(layer, tensor).call();
  }

  @Override
  protected void _free() {
    if (null != input) for (@Nonnull TensorList x : input) x.freeRef();
    layer.freeRef();
    layerDerivative.freeRef();
    if (null != inputDerivative) for (@Nonnull TensorList x : inputDerivative) x.freeRef();
    if (null != output) output.freeRef();
  }

  @Nonnull
  @Override
  public SimpleResult call() {
    TensorList[] inputCopy = Arrays.stream(input).map(x -> x.copy()).toArray(i -> new TensorList[i]);
    inputDerivative = Arrays.stream(inputCopy).map(tensorList -> TensorArray.wrap(tensorList.stream()
        .map(i -> {
          @Nonnull Tensor tensor = new Tensor(i.getDimensions());
          i.freeRef();
          return tensor;
        })
        .toArray(i -> new Tensor[i]))
    ).toArray(i -> new TensorList[i]);
    Result[] inputs = IntStream.range(0, inputCopy.length).mapToObj(i -> {
      return new Result(inputCopy[i], (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
        SimpleListEval.accumulate(inputDerivative[i], data);
      }) {
        @Override
        public boolean isAlive() {
          return true;
        }
      };
    }).toArray(i -> new Result[i]);
    @Nullable final Result eval = layer.eval(inputs);
    for (@Nonnull Result result : inputs) {
      result.freeRef();
    }
    TensorList outputData = eval.getData().copy();
    for (@Nonnull TensorList tensorList : inputCopy) {
      tensorList.freeRef();
    }
    eval.getData().freeRef();
    @Nonnull TensorList tensorList = getFeedback(outputData);
    this.layerDerivative.freeRef();
    this.layerDerivative = new DeltaSet<>();
    eval.accumulate(layerDerivative, tensorList);
    eval.freeRef();
    output = outputData;
    return this;
  }

  /**
   * Get derivative tensor list [ ].
   *
   * @return the tensor list [ ]
   */
  @Override
  public TensorList[] getInputDerivative() {
    return inputDerivative;
  }

  /**
   * Gets feedback.
   *
   * @param data the data
   * @return the feedback
   */
  @Nonnull
  public TensorList getFeedback(@Nonnull final TensorList data) {
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

  public DeltaSet<UUID> getLayerDerivative() {
    return layerDerivative;
  }

  /**
   * Sets key derivative.
   *
   * @param layerDerivative the key derivative
   */
  public void setLayerDerivative(DeltaSet<UUID> layerDerivative) {
    this.layerDerivative = layerDerivative;
  }
}
