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

package com.simiacryptus.mindseye.layers;

import com.simiacryptus.mindseye.layers.NNLayer.ConstNNResult;
import com.simiacryptus.util.ml.Tensor;

import java.util.Arrays;
import java.util.stream.IntStream;

public abstract class NNResult {

  public final TensorList data;

  public NNResult(final Tensor... data) {
    this(new TensorArray(data));
  }

  public NNResult(final TensorList data) {
    super();
    this.data = data;
  }

  /**
   * @param input - An array of inputs, each one of which is a batch for a  given input
   * @return
   */
  public static NNResult[] singleResultArray(Tensor[][] input) {
    return Arrays.stream(input).map((Tensor[] x) -> new ConstNNResult(x)).toArray(i -> new NNResult[i]);
  }
  
  public static NNResult[] singleResultArray(Tensor[] input) {
    return Arrays.stream(input).map((Tensor x) -> new ConstNNResult(x)).toArray(i -> new NNResult[i]);
  }
  
  /**
   * @param batchData - a list examples, ie each sub-array is a single example
   * @return - Returns a result array for NNLayer evaluation
   */
  public static NNResult[] batchResultArray(Tensor[][] batchData) {
    return IntStream.range(0, batchData[0].length).mapToObj(inputIndex -> {
      Tensor[] inputBatch = IntStream.range(0, batchData.length)
                  .mapToObj(trainingExampleId ->batchData[trainingExampleId][inputIndex]).toArray(i -> new Tensor[i]);
      return new ConstNNResult(inputBatch);
    }).toArray(x -> new NNResult[x]);
  }
  
  public final void accumulate(DeltaSet buffer) {
    accumulate(buffer, 1.0);
  }
  
  public final void accumulate(DeltaSet buffer, double value) {
    Tensor[] defaultVector = IntStream.range(0, this.data.length()).mapToObj(i -> {
      assert (Arrays.equals(this.data.get(i).getDimensions(), new int[]{1}));
      return new Tensor(this.data.get(i).getDimensions()).fill(() -> value);
    }).toArray(i -> new Tensor[i]);
    accumulate(buffer, defaultVector);
  }
  
  public abstract void accumulate(DeltaSet buffer, final Tensor[] data);
  
  public abstract boolean isAlive();
  
}
