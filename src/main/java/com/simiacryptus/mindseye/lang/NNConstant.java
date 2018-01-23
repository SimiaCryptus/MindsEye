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
 * A special type of NNResult which ignores backpropigation; it has a constant value.
 */
public final class NNConstant extends NNResult {
  
  /**
   * Instantiates a new Nn constant.
   *
   * @param data the data
   */
  public NNConstant(final Tensor... data) {
    super((final DeltaSet<NNLayer> buffer, final TensorList tensorList) -> {}, data);
  }
  
  /**
   * Instantiates a new Nn constant.
   *
   * @param tensorList the tensor array
   */
  public NNConstant(final TensorList tensorList) {
    super((final DeltaSet<NNLayer> buffer, final TensorList data) -> {}, tensorList);
  }
  
  /**
   * Batch result array nn result [ ].
   *
   * @param input the batch data
   * @return the nn result [ ]
   */
  public static NNResult[] batchResultArray(final Tensor[]... input) {
    return IntStream.range(0, input[0].length).mapToObj(index -> IntStream.range(0, input.length)
                                                                          .mapToObj(id -> input[id][index])
                                                                          .toArray(i -> new Tensor[i]))
                    .map(tensors -> new TensorArray(tensors))
                    .map(tensorArray -> new NNConstant(tensorArray))
                    .toArray(x -> new NNResult[x]);
  }
  
  /**
   * Single result array nn result [ ].
   *
   * @param input the input
   * @return the nn result [ ]
   */
  public static NNResult[] singleResultArray(final Tensor[] input) {
    return Arrays.stream(input).map((final Tensor x) -> new NNConstant(new TensorArray(x))).toArray(i -> new NNResult[i]);
  }
  
  /**
   * Single result array nn result [ ].
   *
   * @param input the input
   * @return the nn result [ ]
   */
  public static NNResult[] singleResultArray(final Tensor[][] input) {
    return Arrays.stream(input).map((final Tensor[] x) -> new NNConstant(new TensorArray(x))).toArray(i -> new NNResult[i]);
  }
  
  @Override
  public boolean isAlive() {
    return false;
  }
  
  
}
