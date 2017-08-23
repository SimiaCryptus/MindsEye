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

import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Tensor;
import com.simiacryptus.util.ml.WeakCachedSupplier;

import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * The type Stochastic array trainable.
 *
 * TODO: Redesign this package. This class is absorbing too many features.
 */
public class StochasticArrayTrainable extends GpuTrainable {
  
  private static final int LOW_MEM_USE = 4 * 1024 * 1024 * 1024;
  private final List<? extends Supplier<Tensor[]>> trainingData;
  private long hash = Util.R.get().nextLong();
  
  /**
   * Instantiates a new Stochastic array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param trainingSize the training size
   */
  public StochasticArrayTrainable(Tensor[][] trainingData, NNLayer network, int trainingSize) {
    this(trainingData, network, trainingSize, trainingSize / CudaExecutionContext.gpuContexts.size());
  }
  
  /**
   * Instantiates a new Stochastic array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param trainingSize the training size
   * @param batchSize    the batch size
   */
  public StochasticArrayTrainable(Tensor[][] trainingData, NNLayer network, int trainingSize, int batchSize) {
    super(network);
    if(0 == trainingData.length) throw new IllegalArgumentException();
    this.trainingData = Arrays.stream(trainingData).map(obj->new WeakCachedSupplier<Tensor[]>(()->obj)).collect(Collectors.toList());
    this.trainingSize = trainingSize;
    resetSampling();
  }
  
  public StochasticArrayTrainable(List<? extends Supplier<Tensor[]>> trainingData, NNLayer network, int trainingSize) {
    this(trainingData, network, trainingSize, trainingSize);
  }
  
  public StochasticArrayTrainable(List<? extends Supplier<Tensor[]>> trainingData, NNLayer network, int trainingSize, int batchSize) {
    super(network);
    if(0 == trainingData.size()) throw new IllegalArgumentException();
    this.trainingData = trainingData;
    this.trainingSize = trainingSize;
    resetSampling();
  }
  
  @Override
  public void resetToFull() {
    this.setSampledData(trainingData.stream().collect(Collectors.toList()));
  }
  
  @Override
  public boolean resetSampling() {
    setHash(Util.R.get().nextLong());
    return true;
  }
  
  /**
   * Gets training size.
   *
   * @return the training size
   */
  public int getTrainingSize() {
    return this.trainingSize;
  }
  
  /**
   * Sets training size.
   *
   * @param trainingSize the training size
   * @return the training size
   */
  public StochasticArrayTrainable setTrainingSize(final int trainingSize) {
    this.trainingSize = trainingSize;
    refreshSampledData();
    return this;
  }
  
  private void setHash(long newValue) {
    if (this.hash == newValue) return;
    this.hash = newValue;
    refreshSampledData();
  }
  
  protected void refreshSampledData() {
    lastPtr = null;
    lastWeights = null;
    assert 0 < trainingData.size();
    assert 0 < getTrainingSize();
    this.setSampledData(trainingData.stream().parallel() //
                           .filter(x->x!=null && x.get()!=null)
                           .sorted(Comparator.comparingLong(y -> System.identityHashCode(y) ^ this.hash)) //
                           .limit(getTrainingSize()) //
                           .collect(Collectors.toList()));
  }
  
}
