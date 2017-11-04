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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.function.WeakCachedSupplier;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * The type Stochastic array trainable.
 * <p>
 */
public class StochasticArrayTrainable extends CachedTrainable<ArrayTrainable> implements StochasticTrainable, TrainableDataMask {
  
  private final List<? extends Supplier<Tensor[]>> trainingData;
  private int trainingSize;
  private long hash = Util.R.get().nextLong();
  
  /**
   * Instantiates a new Stochastic array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param trainingSize the training size
   * @param batchSize    the batch size
   */
  public StochasticArrayTrainable(Tensor[][] trainingData, NNLayer network, int trainingSize, int batchSize) {
    super(new ArrayTrainable(network, batchSize));
    if (0 == trainingData.length) throw new IllegalArgumentException();
    this.trainingData = Arrays.stream(trainingData).map(obj -> new WeakCachedSupplier<Tensor[]>(() -> obj)).collect(Collectors.toList());
    this.trainingSize = trainingSize;
    resetSampling();
  }
  
  /**
   * Instantiates a new Stochastic array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param trainingSize the training size
   */
  public StochasticArrayTrainable(Tensor[][] trainingData, NNLayer network, int trainingSize) {
    this(trainingData, network, trainingSize, trainingSize);
  }
  
  @Override
  public CachedTrainable<? extends Trainable> cached() {
    return this;
  }
  
  /**
   * Instantiates a new Stochastic array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param trainingSize the training size
   */
  public StochasticArrayTrainable(List<? extends Supplier<Tensor[]>> trainingData, NNLayer network, int trainingSize) {
    this(trainingData, network, trainingSize, trainingSize);
  }
  
  /**
   * Instantiates a new Stochastic array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param trainingSize the training size
   * @param batchSize    the batch size
   */
  public StochasticArrayTrainable(List<? extends Supplier<Tensor[]>> trainingData, NNLayer network, int trainingSize, int batchSize) {
    super(new ArrayTrainable(null, network, batchSize));
    if (0 == trainingData.size()) throw new IllegalArgumentException();
    this.trainingData = trainingData;
    this.trainingSize = trainingSize;
    resetSampling();
  }
  
  @Override
  public void resetToFull() {
    getInner().setTrainingData(trainingData.stream().toArray(i->new Tensor[i][]));
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
  @Override
  public int getTrainingSize() {
    return this.trainingSize;
  }
  
  /**
   * Sets training size.
   *
   * @param trainingSize the training size
   * @return the training size
   */
  @Override
  public StochasticTrainable setTrainingSize(final int trainingSize) {
    this.trainingSize = trainingSize;
    refreshSampledData();
    return this;
  }
  
  private void setHash(long newValue) {
    if (this.hash == newValue) return;
    this.hash = newValue;
    refreshSampledData();
  }
  
  /**
   * Refresh sampled data.
   */
  protected void refreshSampledData() {
    assert 0 < trainingData.size();
    getInner().setTrainingData(0 < getTrainingSize() ? (trainingData.stream().parallel() //
                           .filter(x -> x != null && x.get() != null)
                           .sorted(Comparator.comparingLong(y -> System.identityHashCode(y) ^ this.hash)) //
                           .limit(getTrainingSize()).map(x->x.get())
                           .toArray(i->new Tensor[i][])) : trainingData.stream().toArray(i->new Tensor[i][]));
  }
  
}
