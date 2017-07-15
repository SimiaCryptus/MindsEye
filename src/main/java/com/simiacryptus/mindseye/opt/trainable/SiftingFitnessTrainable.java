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
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class SiftingFitnessTrainable implements Trainable {
  
  public static SiftingFitnessTrainable Sqrt(Tensor[][] trainingData, DAGNetwork network, int trainingSize) {
    return Pow(trainingData, network, trainingSize);
  }
  
  public static SiftingFitnessTrainable Pow(Tensor[][] trainingData, DAGNetwork network, int initialTrainingSize) {
    return new SiftingFitnessTrainable(trainingData, network, initialTrainingSize);
  }
  
  private final Tensor[][] trainingData;
  private final DAGNetwork network;
  private long hash = Util.R.get().nextLong();
  private double trainingSize;
  private Tensor[][] sampledData;
  private Tensor[][] holdoverData = new Tensor[][]{};
  private boolean shuffled = false;
  private double holdoverFraction = 0.1;
  
  public SiftingFitnessTrainable(Tensor[][] trainingData, DAGNetwork network, int trainingSize) {
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
    // holdoverData
    holdoverData = IntStream.range(0,result.data.length).mapToObj(x->x)
                           .sorted(Comparator.comparingDouble(x -> -Arrays.stream(result.data[(int) x].getData()).sum()))
                           .map(i->sampledData[i])
                           .limit((long) (sampledData.length * holdoverFraction))
                           .toArray(i -> new Tensor[i][]);
    ScalarStatistics statistics = new ScalarStatistics();
    Arrays.stream(result.data).flatMapToDouble(x-> Arrays.stream(x.getData())).forEach(x->statistics.add(x));
    return new PointSample(deltaSet, stateSet, statistics.getMean());
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
  
  public double getTrainingSize() {
    return this.trainingSize;
  }
  
  public SiftingFitnessTrainable setTrainingSize(final int trainingSize) {
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
    this.sampledData = Stream.concat(
        stream.limit((long) getTrainingSize()-holdoverData.length),
        Arrays.stream(holdoverData)
    ).toArray(i -> new Tensor[i][]);
  }
  
  public SiftingFitnessTrainable setTrainingSize(double trainingSize) {
    this.trainingSize = trainingSize;
    return this;
  }
  
  
  public boolean isShuffled() {
    return shuffled;
  }
  
  public SiftingFitnessTrainable setShuffled(boolean shuffled) {
    this.shuffled = shuffled;
    return this;
  }
  
  public double getHoldoverFraction() {
    return holdoverFraction;
  }
  
  public SiftingFitnessTrainable setHoldoverFraction(double holdoverFraction) {
    this.holdoverFraction = holdoverFraction;
    return this;
  }
}
