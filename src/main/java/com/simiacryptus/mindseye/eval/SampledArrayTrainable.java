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
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Stochastic array trainable.
 */
public class SampledArrayTrainable extends CachedTrainable<ArrayTrainable> implements SampledTrainable, TrainableDataMask {
  
  private final List<? extends Supplier<Tensor[]>> trainingData;
  private int trainingSize;
  private long seed = Util.R.get().nextInt();
  
  /**
   * Instantiates a new Stochastic array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param trainingSize the training size
   * @param batchSize    the batch size
   */
  public SampledArrayTrainable(Tensor[][] trainingData, NNLayer network, int trainingSize, int batchSize) {
    super(new ArrayTrainable(network, batchSize));
    if (0 == trainingData.length) throw new IllegalArgumentException();
    this.trainingData = Arrays.stream(trainingData).map(obj -> new WeakCachedSupplier<Tensor[]>(() -> obj)).collect(Collectors.toList());
    this.trainingSize = trainingSize;
    this.reseed(System.nanoTime());
  }
  
  /**
   * Instantiates a new Stochastic array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param trainingSize the training size
   */
  public SampledArrayTrainable(Tensor[][] trainingData, NNLayer network, int trainingSize) {
    this(trainingData, network, trainingSize, trainingSize);
  }
  
  /**
   * Instantiates a new Stochastic array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param trainingSize the training size
   */
  public SampledArrayTrainable(List<? extends Supplier<Tensor[]>> trainingData, NNLayer network, int trainingSize) {
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
  public SampledArrayTrainable(List<? extends Supplier<Tensor[]>> trainingData, NNLayer network, int trainingSize, int batchSize) {
    super(new ArrayTrainable(null, network, batchSize));
    if (0 == trainingData.size()) throw new IllegalArgumentException();
    this.trainingData = trainingData;
    this.trainingSize = trainingSize;
    this.reseed(System.nanoTime());
  }
  
  @Override
  public boolean reseed(long seed) {
    setSeed(Util.R.get().nextInt());
    getInner().reseed(seed);
    super.reseed(seed);
    return true;
  }
  
  @Override
  public int getTrainingSize() {
    return this.trainingSize;
  }
  
  @Override
  public SampledTrainable setTrainingSize(final int trainingSize) {
    this.trainingSize = trainingSize;
    refreshSampledData();
    return this;
  }
  
  private void setSeed(int newValue) {
    if (this.seed == newValue) return;
    this.seed = newValue;
    refreshSampledData();
  }
  
  /**
   * Refresh sampled data.
   */
  protected void refreshSampledData() {
    assert 0 < trainingData.size();
    Tensor[][] trainingData;
    if (0 < getTrainingSize() && getTrainingSize() < (this.trainingData.size()-1)) {
      Random random = new Random(seed);
      trainingData = IntStream.generate(() -> random.nextInt(this.trainingData.size()))
        .distinct()
        .mapToObj(i->this.trainingData.get(i))
        .filter(x -> x != null && x.get() != null)
        .limit(getTrainingSize()).map(x -> x.get())
        .toArray(i -> new Tensor[i][]);
    }
    else {
      trainingData = this.trainingData.stream()
        .filter(x -> x != null && x.get() != null)
        .limit(getTrainingSize()).map(x -> x.get())
        .toArray(i -> new Tensor[i][]);
    }
    getInner().setTrainingData(trainingData);
  }
  
  @Override
  public SampledCachedTrainable<? extends SampledTrainable> cached() {
    return new SampledCachedTrainable<>(this);
  }
  
}
