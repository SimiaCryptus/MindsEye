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
 * This type handles the data selection part of stochastic gradient descent training. Between each epoch, a "reset"
 * method is called to re-sample the training data and pass it to the localCopy Trainable implementation.
 */
public class SampledArrayTrainable extends TrainableWrapper<ArrayTrainable> implements SampledTrainable, TrainableDataMask {
  
  private final List<? extends Supplier<Tensor[]>> trainingData;
  private int minSamples = 0;
  private long seed = Util.R.get().nextInt();
  private int trainingSize;
  
  /**
   * Instantiates a new Stochastic array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param trainingSize the training size
   */
  public SampledArrayTrainable(final List<? extends Supplier<Tensor[]>> trainingData, final NNLayer network, final int trainingSize) {
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
  public SampledArrayTrainable(final List<? extends Supplier<Tensor[]>> trainingData, final NNLayer network, final int trainingSize, final int batchSize) {
    super(new ArrayTrainable(null, network, batchSize));
    if (0 == trainingData.size()) throw new IllegalArgumentException();
    this.trainingData = trainingData;
    this.trainingSize = trainingSize;
    reseed(System.nanoTime());
  }
  
  /**
   * Instantiates a new Stochastic array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param trainingSize the training size
   */
  public SampledArrayTrainable(final Tensor[][] trainingData, final NNLayer network, final int trainingSize) {
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
  public SampledArrayTrainable(final Tensor[][] trainingData, final NNLayer network, final int trainingSize, final int batchSize) {
    super(new ArrayTrainable(network, batchSize));
    if (0 == trainingData.length) throw new IllegalArgumentException();
    this.trainingData = Arrays.stream(trainingData).map(obj -> new WeakCachedSupplier<>(() -> obj)).collect(Collectors.toList());
    this.trainingSize = trainingSize;
    reseed(System.nanoTime());
  }
  
  @Override
  public SampledCachedTrainable<? extends SampledTrainable> cached() {
    return new SampledCachedTrainable<>(this);
  }
  
  /**
   * Gets min samples.
   *
   * @return the min samples
   */
  public int getMinSamples() {
    return minSamples;
  }
  
  /**
   * Sets min samples.
   *
   * @param minSamples the min samples
   * @return the min samples
   */
  public SampledArrayTrainable setMinSamples(final int minSamples) {
    this.minSamples = minSamples;
    return this;
  }
  
  @Override
  public int getTrainingSize() {
    return Math.max(minSamples, Math.min(trainingData.size(), trainingSize));
  }
  
  /**
   * Refresh sampled data.
   */
  protected void refreshSampledData() {
    assert 0 < trainingData.size();
    Tensor[][] trainingData;
    if (0 < getTrainingSize() && getTrainingSize() < this.trainingData.size() - 1) {
      final Random random = new Random(seed);
      trainingData = IntStream.generate(() -> random.nextInt(this.trainingData.size()))
                              .distinct()
                              .mapToObj(i -> this.trainingData.get(i))
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
  public boolean reseed(final long seed) {
    setSeed(Util.R.get().nextInt());
    getInner().reseed(seed);
    super.reseed(seed);
    return true;
  }
  
  private void setSeed(final int newValue) {
    if (seed == newValue) return;
    seed = newValue;
    refreshSampledData();
  }
  
  @Override
  public SampledTrainable setTrainingSize(final int trainingSize) {
    this.trainingSize = trainingSize;
    refreshSampledData();
    return this;
  }
}
