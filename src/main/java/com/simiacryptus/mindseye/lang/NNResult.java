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

package com.simiacryptus.mindseye.lang;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * This type holds the result from a NNLayer evaluation.
 * It holds the result and a callback mechanism to evaluate the derivatives.
 */
public abstract class NNResult {
  
  /**
   * The Data.
   */
  protected final TensorList data;
  
  /**
   * Instantiates a new Nn result.
   *
   * @param data the data
   */
  public NNResult(final Tensor... data) {
    this(new TensorArray(data));
  }
  
  /**
   * Instantiates a new Nn result.
   *
   * @param data the data
   */
  public NNResult(final TensorList data) {
    super();
    this.data = data;
  }
  
  /**
   * Batch result array nn result [ ].
   *
   * @param batchData the batch data
   * @return the nn result [ ]
   */
  public static NNResult[] batchResultArray(final Tensor[][] batchData) {
    return IntStream.range(0, batchData[0].length).mapToObj(inputIndex ->
      new NNConstant(new TensorArray(IntStream.range(0, batchData.length).mapToObj(trainingExampleId ->
        batchData[trainingExampleId][inputIndex]
      ).toArray(i -> new Tensor[i])))).toArray(x -> new NNResult[x]);
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
  
  /**
   * Accumulate.
   *
   * @param buffer the buffer
   */
  public final void accumulate(final DeltaSet<NNLayer> buffer) {
    accumulate(buffer, 1.0);
  }
  
  /**
   * Accumulate.
   *
   * @param buffer the buffer
   * @param value  the value
   */
  public final void accumulate(final DeltaSet<NNLayer> buffer, final double value) {
    final Tensor[] defaultVector = getData().stream().map(t -> t.map(v -> value)).toArray(i -> new Tensor[i]);
    accumulate(buffer, new TensorArray(defaultVector));
  }
  
  /**
   * Accumulate.
   *
   * @param buffer the buffer
   * @param data   the data
   */
  public abstract void accumulate(DeltaSet<NNLayer> buffer, final TensorList data);
  
  /**
   * Gets data.
   *
   * @return the data
   */
  public TensorList getData() {
    return data;
  }
  
  /**
   * Is alive boolean.
   *
   * @return the boolean
   */
  public abstract boolean isAlive();
  
}
