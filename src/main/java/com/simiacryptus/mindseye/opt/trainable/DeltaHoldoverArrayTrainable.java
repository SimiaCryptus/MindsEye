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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.function.WeakCachedSupplier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Stochastic array trainable.
 * <p>
 * TODO: Redesign this package. This class is absorbing too many features.
 */
public class DeltaHoldoverArrayTrainable extends GpuTrainable {
  
  private static final int LOW_MEM_USE = 4 * 1024 * 1024 * 1024;
  private final List<? extends Supplier<Tensor[]>> trainingData;
  private final WrappingLayer wrappingLayer;
  private int trainingSize;
  private long hash = Util.R.get().nextLong();
  private double holdoverFraction = 0.5;
  
  /**
   * Instantiates a new Stochastic array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param trainingSize the training size
   */
  public DeltaHoldoverArrayTrainable(Tensor[][] trainingData, NNLayer network, int trainingSize) {
    super(new WrappingLayer(network));
    this.wrappingLayer = new WrappingLayer(network);
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
  public DeltaHoldoverArrayTrainable(List<? extends Supplier<Tensor[]>> trainingData, NNLayer network, int trainingSize) {
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
  public DeltaHoldoverArrayTrainable(List<? extends Supplier<Tensor[]>> trainingData, NNLayer network, int trainingSize, int batchSize) {
    super(new WrappingLayer(network));
    this.wrappingLayer = (WrappingLayer) network;
    if (0 == trainingData.size()) throw new IllegalArgumentException();
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
  public DeltaHoldoverArrayTrainable setTrainingSize(final int trainingSize) {
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
    if(0 >= getTrainingSize()) {
      setSampledData(trainingData);
    } else {
      List<Tensor[]> holdover;
      if(wrappingLayer.firstResult != null && wrappingLayer.lastResult != null && wrappingLayer.firstResult != wrappingLayer.lastResult) {
        holdover = IntStream.range(0, sampledData.size()).mapToObj(x -> x)
                                    .sorted(Comparator.comparingDouble(x -> {
                                      double currentData = Arrays.stream(wrappingLayer.firstResult.getData().get((int) x).getData()).sum();
                                      double latestData = Arrays.stream(wrappingLayer.lastResult.getData().get((int) x).getData()).sum();
                                      return (latestData - currentData);
                                    }))
                                    .map(i -> sampledData.get(i))
                                    .limit((long)(getTrainingSize() * holdoverFraction))
                                    .collect(Collectors.toList());
        
      } else {
        holdover = new ArrayList<>();
      }
      List<Tensor[]> extra = trainingData.stream().filter(x -> x != null)
                               .map(x -> x.get())
                               .filter(x -> x != null)
                               .filter(x -> !holdover.contains(x))
                               .sorted(Comparator.comparingLong(y -> System.identityHashCode(y) ^ this.hash)) //
                               .limit(getTrainingSize() - holdover.size()) //
                               .collect(Collectors.toList());
      ArrayList<Tensor[]> concat = new ArrayList<>();
      concat.addAll(holdover);
      concat.addAll(extra);
      setData(concat);
    }
  }
  
  private static class WrappingLayer extends NNLayer {
    NNLayer inner;
    NNResult lastResult;
    NNResult firstResult;
    
    public WrappingLayer(NNLayer network) {
      this.inner = network;
    }
    
    @Override
    public NNResult eval(NNExecutionContext nncontext, NNResult[] array) {
      NNResult result = inner.eval(nncontext, array);
      this.lastResult = result;
      if (this.firstResult == null) this.firstResult = result;
      return result;
    }
    
    @Override
    public JsonObject getJson() {
      return inner.getJson();
    }
    
    @Override
    public List<double[]> state() {
      return inner.state();
    }
  }
}
