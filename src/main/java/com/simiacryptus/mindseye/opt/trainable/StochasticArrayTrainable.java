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
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Tensor;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * The type Stochastic array trainable.
 */
public class StochasticArrayTrainable implements Trainable {
  
  private final Tensor[][] trainingData;
  private final NNLayer network;
  private long hash = Util.R.get().nextLong();
  private int trainingSize = Integer.MAX_VALUE;
  private int batchSize = Integer.MAX_VALUE;
  private Tensor[][] sampledData;
  
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
    this.trainingData = trainingData;
    this.network = network;
    this.trainingSize = trainingSize;
    this.batchSize = batchSize;
    resetSampling();
  }
  
  @Override
  public PointSample measure() {
    return Lists.partition(Arrays.asList(sampledData), batchSize).stream().parallel().map(trainingData->{
      PointSample pointSample = evalSubsample(trainingData);
      System.gc();
      return pointSample;
    }).reduce((a,b)->new PointSample(a.delta.add(b.delta), a.weights,a.value + b.value)).get();
  }
  
  /**
   * Eval subsample point sample.
   *
   * @param trainingData the training data
   * @return the point sample
   */
  protected PointSample evalSubsample(List<Tensor[]> trainingData) {
    NNResult[] input = NNResult.batchResultArray(trainingData.toArray(new Tensor[][]{}));
    return CuDNN.gpuContexts.with(nncontext->{
      NNResult result = network.eval(nncontext, input);
      DeltaSet deltaSet = new DeltaSet();
      result.accumulate(deltaSet);
      assert (deltaSet.vector().stream().allMatch(x -> Arrays.stream(x.delta).allMatch(Double::isFinite)));
      DeltaSet stateBackup = new DeltaSet();
      deltaSet.map.forEach((layer, layerDelta) -> {
        stateBackup.get(layer, layerDelta.target).accumulate(layerDelta.target);
      });
      assert (stateBackup.vector().stream().allMatch(x -> Arrays.stream(x.delta).allMatch(Double::isFinite)));
      assert (result.data.stream().allMatch(x -> x.dim() == 1));
      assert (result.data.stream().allMatch(x -> Arrays.stream(x.getData()).allMatch(Double::isFinite)));
      double meanValue = result.data.stream().mapToDouble(x -> x.getData()[0]).sum();
      return new PointSample(deltaSet.scale(1.0/trainingSize), stateBackup, meanValue / trainingSize);
    });
  }

  @Override
  public void resetToFull() {
    sampledData = trainingData;
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
    final Tensor[][] rawData = trainingData;
    assert 0 < rawData.length;
    assert 0 < getTrainingSize();
    this.sampledData = Arrays.stream(rawData).parallel() //
                           .sorted(Comparator.comparingLong(y -> System.identityHashCode(y) ^ this.hash)) //
                           .limit(getTrainingSize()) //
                           .toArray(i -> new Tensor[i][]);
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
}
