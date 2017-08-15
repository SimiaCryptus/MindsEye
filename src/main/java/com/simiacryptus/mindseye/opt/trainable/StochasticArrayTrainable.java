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
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.layers.cudnn.CuDNN;
import com.simiacryptus.mindseye.layers.cudnn.CudaPtr;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Tensor;
import com.simiacryptus.util.ml.WeakCachedSupplier;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * The type Stochastic array trainable.
 *
 * TODO: Redesign this package. This class is absorbing too many features.
 */
public class StochasticArrayTrainable implements Trainable {
  
  private static final int LOW_MEM_USE = 4 * 1024 * 1024 * 1024;
  private final List<? extends Supplier<Tensor[]>> trainingData;
  private final NNLayer network;
  private long hash = Util.R.get().nextLong();
  private int trainingSize = Integer.MAX_VALUE;
  private int batchSize = Integer.MAX_VALUE;
  private boolean gcEachIteration = true;
  private List<NNResult[]> trainingNNResultArrays;
  private List<? extends Supplier<Tensor[]>> sampledData;
  
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
  
  /**
   * Instantiates a new Stochastic array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param trainingSize the training size
   * @param batchSize    the batch size
   */
  public StochasticArrayTrainable(Tensor[][] trainingData, NNLayer network, int trainingSize, int batchSize) {
    if(0 == trainingData.length) throw new IllegalArgumentException();
    this.trainingData = Arrays.stream(trainingData).map(obj->new WeakCachedSupplier<Tensor[]>(()->obj)).collect(Collectors.toList());
    this.network = network;
    this.trainingSize = trainingSize;
    this.batchSize = batchSize;
    resetSampling();
  }
  
  public StochasticArrayTrainable(List<? extends Supplier<Tensor[]>> trainingData, NNLayer network, int trainingSize) {
    this(trainingData, network, trainingSize, trainingSize);
  }
  
  public StochasticArrayTrainable(List<? extends Supplier<Tensor[]>> trainingData, NNLayer network, int trainingSize, int batchSize) {
    if(0 == trainingData.size()) throw new IllegalArgumentException();
    this.trainingData = trainingData;
    this.network = network;
    this.trainingSize = trainingSize;
    this.batchSize = batchSize;
    resetSampling();
  }
  
  @Override
  public PointSample measure() {
    try {
      PointSample result = trainingNNResultArrays.stream()
        .map(this::evalSubsample)
        .reduce((a, b) -> new PointSample(a.delta.add(b.delta), a.weights, a.value + b.value))
        .get();
      Map<Integer, Long> peakMemory = CudaPtr.METRICS.asMap().entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue().peakMemory.getAndSet(0)));
      if(trainingNNResultArrays.size() < CuDNN.gpuContexts.size()) {
        batchSize = batchSize / 2;
      } else if(trainingNNResultArrays.size() > 2 * CuDNN.gpuContexts.size()) {
        double highestMemUse = peakMemory.values().stream().mapToDouble(x -> x).max().getAsDouble();
        if(highestMemUse < LOW_MEM_USE) {
          batchSize = batchSize * 2;
        }
      }
      if(gcEachIteration) cleanMemory();
      return result;
    } catch (Throwable t) {
      if(isOom(t) && batchSize > 1) {
        batchSize = batchSize / 2;
        regenerateNNResultInputs();
        cleanMemory();
        return measure();
      } else throw new RuntimeException("Failed executing " + this,t);
    }
  }
  
  public void cleanMemory() {
    Tensor.clear();
    System.gc();
    System.runFinalization();
  }
  
  private static boolean isOom(Throwable t) {
    if(t instanceof OutOfMemoryError) return true;
    if(null != t.getCause() && t != t.getCause()) return isOom(t.getCause());
    return false;
  }
  
  private PointSample evalSubsample(NNResult[] input) {
    return CuDNN.gpuContexts.map(nncontext->{
      NNResult result = network.eval(nncontext, input);
      DeltaSet deltaSet = new DeltaSet();
      result.accumulate(deltaSet);
      assert (deltaSet.vector().stream().allMatch(x -> Arrays.stream(x.getDelta()).allMatch(Double::isFinite)));
      DeltaSet stateBackup = new DeltaSet();
      deltaSet.map.forEach((layer, layerDelta) -> {
        stateBackup.get(layer, layerDelta.target).accumulate(layerDelta.target);
      });
      assert (stateBackup.vector().stream().allMatch(x -> Arrays.stream(x.getDelta()).allMatch(Double::isFinite)));
      assert (result.getData().stream().allMatch(x -> x.dim() == 1));
      assert (result.getData().stream().allMatch(x -> Arrays.stream(x.getData()).allMatch(Double::isFinite)));
      double meanValue = result.getData().stream().mapToDouble(x -> x.getData()[0]).sum();
      return new PointSample(deltaSet.scale(1.0/trainingSize), stateBackup, meanValue / trainingSize);
    });
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
  
  private void refreshSampledData() {
    assert 0 < trainingData.size();
    assert 0 < getTrainingSize();
    this.setSampledData(trainingData.stream().parallel() //
                           .sorted(Comparator.comparingLong(y -> System.identityHashCode(y) ^ this.hash)) //
                           .filter(x->x.get()!=null)
                           .limit(getTrainingSize()) //
                           .collect(Collectors.toList()));
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
   * Sets batch size.
   *
   * @param batchSize the batch size
   */
  public void setBatchSize(int batchSize) {
    this.batchSize = batchSize;
  }
  
  public boolean isGcEachIteration() {
    return gcEachIteration;
  }
  
  public void setGcEachIteration(boolean gcEachIteration) {
    this.gcEachIteration = gcEachIteration;
  }
  
  protected void setSampledData(List<? extends Supplier<Tensor[]>> sampledData) {
    this.sampledData = sampledData;
    regenerateNNResultInputs();
  
  }
  
  private void regenerateNNResultInputs() {
    this.trainingNNResultArrays = Lists.partition(this.sampledData, batchSize).stream().parallel()
                                                .map(xx -> NNResult.batchResultArray(xx.stream().parallel().map(x->x.get()).toArray(i->new Tensor[i][])))
                                                .collect(Collectors.toList());
  }
}
