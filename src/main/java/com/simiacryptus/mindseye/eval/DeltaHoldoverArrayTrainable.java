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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.function.WeakCachedSupplier;

import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Experimental. The idea behind this class is to track for each row the change in value for an objective function over
 * the course of a single training epoch. Rows which are observed to have a high delta are inferred to be "interesting"
 * rows, subject to retention when re-sampling training data between epochs.
 */
public class DeltaHoldoverArrayTrainable extends GpuTrainable {
  
  private final double holdoverFraction = 0.5;
  private final List<? extends Supplier<Tensor[]>> trainingData;
  private final WrappingLayer wrappingLayer;
  private long hash = Util.R.get().nextLong();
  private int trainingSize;
  
  /**
   * Instantiates a new Delta holdover array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param trainingSize the training size
   */
  public DeltaHoldoverArrayTrainable(final List<? extends Supplier<Tensor[]>> trainingData, final NNLayer network, final int trainingSize) {
    this(trainingData, network, trainingSize, trainingSize);
  }
  
  /**
   * Instantiates a new Delta holdover array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param trainingSize the training size
   * @param batchSize    the batch size
   */
  public DeltaHoldoverArrayTrainable(final List<? extends Supplier<Tensor[]>> trainingData, final NNLayer network, final int trainingSize, final int batchSize) {
    super(new WrappingLayer(network));
    wrappingLayer = (WrappingLayer) network;
    if (0 == trainingData.size()) throw new IllegalArgumentException();
    this.trainingData = trainingData;
    this.trainingSize = trainingSize;
    reseed(System.nanoTime());
  }
  
  /**
   * Instantiates a new Delta holdover array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param trainingSize the training size
   */
  public DeltaHoldoverArrayTrainable(final Tensor[][] trainingData, final NNLayer network, final int trainingSize) {
    super(new WrappingLayer(network));
    wrappingLayer = new WrappingLayer(network);
    if (0 == trainingData.length) throw new IllegalArgumentException();
    this.trainingData = Arrays.stream(trainingData).map(obj -> new WeakCachedSupplier<>(() -> obj)).collect(Collectors.toList());
    this.trainingSize = trainingSize;
    reseed(System.nanoTime());
  }
  
  /**
   * Gets training size.
   *
   * @return the training size
   */
  public int getTrainingSize() {
    return trainingSize;
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
  
  /**
   * Refresh sampled data.
   */
  protected void refreshSampledData() {
    assert 0 < trainingData.size();
    if (0 >= getTrainingSize()) {
      setDataSupplier(trainingData);
    }
    else {
      List<Tensor[]> holdover;
      if (wrappingLayer.firstResult != null && wrappingLayer.lastResult != null && wrappingLayer.firstResult != wrappingLayer.lastResult) {
        holdover = IntStream.range(0, data.size()).mapToObj(x -> x)
                            .sorted(Comparator.comparingDouble(x -> {
                              final double currentData = Arrays.stream(wrappingLayer.firstResult.getData().get(x).getData()).sum();
                              final double latestData = Arrays.stream(wrappingLayer.lastResult.getData().get(x).getData()).sum();
                              return latestData - currentData;
                            }))
                            .map(i -> data.get(i))
                            .limit((long) (getTrainingSize() * holdoverFraction))
                            .collect(Collectors.toList());
        
      }
      else {
        holdover = new ArrayList<>();
      }
      final List<Tensor[]> extra = trainingData.stream().filter(x -> x != null)
                                               .map(x -> x.get())
                                               .filter(x -> x != null)
                                               .filter(x -> !holdover.contains(x))
                                               .sorted(Comparator.comparingLong(y -> System.identityHashCode(y) ^ hash)) //
                                               .limit(getTrainingSize() - holdover.size()) //
                                               .collect(Collectors.toList());
      final ArrayList<Tensor[]> concat = new ArrayList<>();
      concat.addAll(holdover);
      concat.addAll(extra);
      setData(concat);
    }
  }
  
  @Override
  public boolean reseed(final long seed) {
    setHash(Util.R.get().nextLong());
    return true;
  }
  
  private void setHash(final long newValue) {
    if (hash == newValue) return;
    hash = newValue;
    refreshSampledData();
  }
  
  @SuppressWarnings("serial")
  private static class WrappingLayer extends NNLayer {
    /**
     * The First result.
     */
    NNResult firstResult;
    /**
     * The Inner.
     */
    NNLayer inner;
    /**
     * The Last result.
     */
    NNResult lastResult;
  
    /**
     * Instantiates a new Wrapping layer.
     *
     * @param network the network
     */
    public WrappingLayer(final NNLayer network) {
      inner = network;
    }
    
    @Override
    public NNResult eval(final NNExecutionContext nncontext, final NNResult... array) {
      final NNResult result = inner.eval(nncontext, array);
      lastResult = result;
      if (firstResult == null) {
        firstResult = result;
      }
      return result;
    }
    
    @Override
    public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
      return inner.getJson(resources, dataSerializer);
    }
    
    @Override
    public List<double[]> state() {
      return inner.state();
    }
  }
}
