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

package com.simiacryptus.mindseye.lang;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * A special type of Result which ignores backpropigation; it has a constant value.
 */
public final class ConstantResult extends Result {
  
  /**
   * Instantiates a new Nn constant.
   *
   * @param data the data
   */
  public ConstantResult(final Tensor... data) {
    super(TensorArray.create(data), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList tensorList) -> {});
  }
  
  /**
   * Instantiates a new Nn constant.
   *
   * @param tensorList the tensor array
   */
  public ConstantResult(final TensorList tensorList) {
    super(tensorList, (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList data) -> {});
  }
  
  /**
   * Batch result array nn result [ ].
   *
   * @param input the batch data
   * @return the nn result [ ]
   */
  public static Result[] batchResultArray(@javax.annotation.Nonnull final Tensor[]... input) {
    if (null == input) throw new IllegalArgumentException();
    return IntStream.range(0, input[0].length).mapToObj(index -> IntStream.range(0, input.length)
      .mapToObj(id -> input[id][index])
      .toArray(i -> new Tensor[i]))
      .map(tensors -> TensorArray.create(tensors))
      .map(tensorArray -> new ConstantResult(tensorArray))
      .toArray(x -> new Result[x]);
  }
  
  /**
   * Single result array nn result [ ].
   *
   * @param input the input
   * @return the nn result [ ]
   */
  public static Result[] singleResultArray(@javax.annotation.Nonnull final Tensor[] input) {
    return Arrays.stream(input).map((@javax.annotation.Nonnull final Tensor x) -> new ConstantResult(TensorArray.create(x))).toArray(i -> new Result[i]);
  }
  
  /**
   * Single result array nn result [ ].
   *
   * @param input the input
   * @return the nn result [ ]
   */
  public static Result[] singleResultArray(@javax.annotation.Nonnull final Tensor[][] input) {
    return Arrays.stream(input).map((@javax.annotation.Nonnull final Tensor[] x) -> new ConstantResult(TensorArray.create(x))).toArray(i -> new Result[i]);
  }
  
  @Override
  public boolean isAlive() {
    return false;
  }
  
  
}
