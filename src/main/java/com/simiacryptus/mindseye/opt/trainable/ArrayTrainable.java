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

package com.simiacryptus.mindseye.opt.trainable;

import com.google.common.collect.Lists;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public abstract class ArrayTrainable extends CachedTrainable<GpuTrainable> {
  
  protected final int batchSize;
  
  public ArrayTrainable(NNLayer network, int batchSize) {
    super(new GpuTrainable(network));
    this.batchSize = batchSize;
  }
  
  @Override
  public PointSample measure() {
    List<List<Tensor[]>> collection = batchSize < getTrainingData().length ?
                                        Lists.partition(Arrays.asList(getTrainingData()), batchSize)
                                        : Arrays.asList(Arrays.asList(getTrainingData()));
    return collection.stream().map(trainingData -> {
      getInner().setSampledData(trainingData.stream().map(x->(Supplier<Tensor[]>)(()->x)).collect(Collectors.toList()));
      return getInner().measure();
    }).reduce((a, b) -> new PointSample(a.delta.add(b.delta), a.weights, a.value + b.value)).get();
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
  
  public abstract Tensor[][] getTrainingData();
}
