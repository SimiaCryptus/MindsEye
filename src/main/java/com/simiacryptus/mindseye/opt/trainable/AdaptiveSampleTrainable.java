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

import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.util.ScalarStatistics;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Tensor;

import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.Stream;

public class AdaptiveSampleTrainable implements Trainable {
  
  private final Tensor[][] trainingData;
  private final DAGNetwork network;
  private long hash = Util.R.get().nextLong();
  private int trainingSize = Integer.MAX_VALUE;
  private int trainingSizeMax = Integer.MAX_VALUE;
  private int trainingSizeMin = 1;
  private Tensor[][] sampledData;
  private double alpha = alpha = 0.1;
  private double beta = 0.01;
  private boolean shuffled = true;
  
  public AdaptiveSampleTrainable(Tensor[][] trainingData, DAGNetwork network, int trainingSize) {
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
    assert (Arrays.stream(result.data).allMatch(x -> x.dim() == 1));
    ScalarStatistics statistics = new ScalarStatistics();
    Arrays.stream(result.data).forEach(x->statistics.add(x.getData()));
    double mean = statistics.getMean();
    double stdDev = statistics.getStdDev();
    double samplingError = mean / Math.sqrt(statistics.getCount());
    System.out.println(String.format("%s %s %s", statistics.getCount(), mean, stdDev, samplingError));
    if(samplingError > alpha * stdDev) trainingSize *= 2;
    else if(samplingError < beta * stdDev) trainingSize /= 2;
    return new PointSample(deltaSet, stateSet, mean);
  }
  
  @Override
  public void resetToFull() {
    sampledData = trainingData;
  }
  
  @Override
  public void resetSampling() {
    setHash(Util.R.get().nextLong());
  }
  
  public int getTrainingSize() {
    return this.trainingSize;
  }
  
  public AdaptiveSampleTrainable setTrainingSize(final int trainingSize) {
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
    Stream<Tensor[]> stream = Arrays.stream(rawData).parallel();
    if(shuffled) {
      stream = stream.sorted(Comparator.comparingLong(y -> System.identityHashCode(y) ^ this.hash));
    }
    this.sampledData = stream.limit(Math.min(Math.max(getTrainingSize(),trainingSizeMin), trainingSizeMax)) //
                           .toArray(i -> new Tensor[i][]);
  }
  
  public int getTrainingSizeMax() {
    return trainingSizeMax;
  }
  
  public AdaptiveSampleTrainable setTrainingSizeMax(int trainingSizeMax) {
    this.trainingSizeMax = trainingSizeMax;
    return this;
  }
  
  public int getTrainingSizeMin() {
    return trainingSizeMin;
  }
  
  public AdaptiveSampleTrainable setTrainingSizeMin(int trainingSizeMin) {
    this.trainingSizeMin = trainingSizeMin;
    return this;
  }
  
  public double getAlpha() {
    return alpha;
  }
  
  public AdaptiveSampleTrainable setAlpha(double alpha) {
    this.alpha = alpha;
    return this;
  }
  
  public double getBeta() {
    return beta;
  }
  
  public AdaptiveSampleTrainable setBeta(double beta) {
    this.beta = beta;
    return this;
  }
  
  public boolean isShuffled() {
    return shuffled;
  }
  
  public AdaptiveSampleTrainable setShuffled(boolean shuffled) {
    this.shuffled = shuffled;
    return this;
  }
}
