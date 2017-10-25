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

package com.simiacryptus.mindseye.eval;

import com.google.common.collect.Lists;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;

import java.util.Arrays;
import java.util.List;

/**
 * The type Array trainable.
 */
public abstract class ArrayTrainable extends CachedTrainable<GpuTrainable> {
  
  /**
   * The Batch size.
   */
  protected final int batchSize;
  
  /**
   * Instantiates a new Array trainable.
   *
   * @param network   the network
   * @param batchSize the batch size
   */
  public ArrayTrainable(NNLayer network, int batchSize) {
    super(new GpuTrainable(network));
    this.batchSize = batchSize;
  }
  
  @Override
  public PointSample measure() {
    List<Tensor[]> tensors = Arrays.asList(getTrainingData());
    if(batchSize < tensors.size()) {
      List<List<Tensor[]>> collection = Lists.partition(tensors, batchSize);
      return collection.stream().map(trainingData -> {
        if(batchSize < trainingData.size()) {
          throw new RuntimeException();
        }
        return getInner().setData(trainingData).measure();
      }).reduce((a, b) -> a.add(b)).get();
    } else {
      return getInner().setData(tensors).measure();
    }
  }
  
  @Override
  public void resetToFull() {
  }
  
  /**
   * Gets batch size.
   *
   * @return the batch size
   */
  public int getBatchSize() {
    return batchSize;
  }
  
  /**
   * Get training data tensor [ ] [ ].
   *
   * @return the tensor [ ] [ ]
   */
  public abstract Tensor[][] getTrainingData();
}
