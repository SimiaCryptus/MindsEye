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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.NNLayer;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * The type Array trainable.
 */
public class ArrayTrainable extends CachedTrainable<GpuTrainable> {
  
  private final Tensor[][] trainingData;
  private final int batchSize;
  
  /**
   * Instantiates a new Array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   */
  public ArrayTrainable(Tensor[][] trainingData, NNLayer network) {
    this(trainingData, network, trainingData.length);
  }
  
  /**
   * Instantiates a new Array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param batchSize    the batch size
   */
  public ArrayTrainable(Tensor[][] trainingData, NNLayer network, int batchSize) {
    super(new GpuTrainable(network));
    if (0 == trainingData.length) throw new IllegalArgumentException();
    this.trainingData = trainingData;
    this.batchSize = batchSize;
    resetSampling();
  }
  
  @Override
  public PointSample measure() {
    List<List<Tensor[]>> collection = batchSize < trainingData.length ?
                                        Lists.partition(Arrays.asList(trainingData), batchSize)
                                        : Arrays.asList(Arrays.asList(trainingData));
    return collection.stream().map(trainingData -> {
      inner.setSampledData(trainingData.stream().map(x->(Supplier<Tensor[]>)(()->x)).collect(Collectors.toList()));
      return inner.measure();
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
  
}
