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
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Tensor;

import java.util.Arrays;
import java.util.Comparator;

public class StochasticArrayTrainable implements Trainable {
  
  private final Tensor[][] trainingData;
  private final NNLayer network;
  private long hash = Util.R.get().nextLong();
  private int trainingSize = Integer.MAX_VALUE;
  private Tensor[][] sampledData;
  
  public StochasticArrayTrainable(Tensor[][] trainingData, NNLayer network, int trainingSize) {
    this.trainingData = trainingData;
    this.network = network;
    this.trainingSize = trainingSize;
    resetSampling();
  }
  
  @Override
  public PointSample measure() {
    NNResult[] input = NNResult.batchResultArray(sampledData);
    NNResult result = network.eval(input);
    DeltaSet deltaSet = new DeltaSet();
    result.accumulate(deltaSet);
    DeltaSet stateSet = new DeltaSet();
    deltaSet.map.forEach((layer, layerDelta) -> {
      stateSet.get(layer, layerDelta.target).accumulate(layerDelta.target);
    });
    assert (result.data.stream().allMatch(x -> x.dim() == 1));
    double meanValue = result.data.stream().mapToDouble(x -> x.getData()[0]).average().getAsDouble();
    return new PointSample(deltaSet, stateSet, meanValue);
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
  
  public int getTrainingSize() {
    return this.trainingSize;
  }
  
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
}
