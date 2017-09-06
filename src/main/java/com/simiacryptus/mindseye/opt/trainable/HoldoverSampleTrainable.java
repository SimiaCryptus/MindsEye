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

import com.simiacryptus.mindseye.data.Tensor;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.util.PercentileStatistics;
import com.simiacryptus.util.ScalarStatistics;
import com.simiacryptus.util.Util;

import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Holdover sample trainable.
 */
public class HoldoverSampleTrainable implements Trainable {
  
  /**
   * Sqrt holdover sample trainable.
   *
   * @param trainingData    the training data
   * @param network         the network
   * @param trainingSize    the training size
   * @param initialIncrease the initial increase
   * @return the holdover sample trainable
   */
  public static HoldoverSampleTrainable Sqrt(Tensor[][] trainingData, DAGNetwork network, int trainingSize, double initialIncrease) {
    return Pow(trainingData, network, trainingSize, initialIncrease, -0.5);
  }
  
  /**
   * Pow holdover sample trainable.
   *
   * @param trainingData        the training data
   * @param network             the network
   * @param initialTrainingSize the initial training size
   * @param initialIncrease     the initial increase
   * @param powerRelation       the power relation
   * @return the holdover sample trainable
   */
  public static HoldoverSampleTrainable Pow(Tensor[][] trainingData, DAGNetwork network, int initialTrainingSize, double initialIncrease, double powerRelation) {
    return new HoldoverSampleTrainable(trainingData, network, initialTrainingSize, initialIncrease / Math.pow(initialTrainingSize, powerRelation)).setIncreasePower(powerRelation);
  }
  
  private final Tensor[][] trainingData;
  private final DAGNetwork network;
  private long hash = Util.R.get().nextLong();
  private double trainingSize = Integer.MAX_VALUE;
  private int trainingSizeMax = Integer.MAX_VALUE;
  private int trainingSizeMin = 1;
  private Tensor[][] sampledData;
  private Tensor[][] holdoverData = new Tensor[][]{};
  private boolean shuffled = false;
  private double increaseMultiplier = 0;
  private double increasePower = 0;
  private double holdoverFraction = 0.1;
  
  /**
   * Instantiates a new Holdover sample trainable.
   *
   * @param trainingData       the training data
   * @param network            the network
   * @param trainingSize       the training size
   * @param increaseMultiplier the increase multiplier
   */
  public HoldoverSampleTrainable(Tensor[][] trainingData, DAGNetwork network, int trainingSize, double increaseMultiplier) {
    this.trainingData = trainingData;
    this.network = network;
    this.trainingSize = trainingSize;
    this.increaseMultiplier = increaseMultiplier;
    resetSampling();
  }
  
  @Override
  public Trainable.PointSample measure() {
    NNResult[] input = NNResult.batchResultArray(sampledData);
    NNResult result = network.eval(new NNLayer.NNExecutionContext() {
    }, input);
    DeltaSet deltaSet = new DeltaSet();
    result.accumulate(deltaSet);
    DeltaSet stateSet = new DeltaSet();
    deltaSet.map.forEach((layer, layerDelta) -> {
      stateSet.get(layer, layerDelta.target).accumulate(layerDelta.target);
    });
    assert (result.getData().stream().allMatch(x -> x.dim() == 1));
    // holdoverData
    holdoverData = IntStream.range(0, result.getData().length()).mapToObj(x -> x)
                     .sorted(Comparator.comparingDouble(x -> -Arrays.stream(result.getData().get((int) x).getData()).sum()))
                     .map(i -> sampledData[i])
                     .limit((long) (sampledData.length * holdoverFraction))
                     .toArray(i -> new Tensor[i][]);
    ScalarStatistics statistics = new PercentileStatistics();
    result.getData().stream().flatMapToDouble(x -> Arrays.stream(x.getData())).forEach(x -> statistics.add(x));
    return new Trainable.PointSample(deltaSet, stateSet, statistics.getMean());
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
  public double getTrainingSize() {
    return this.trainingSize;
  }
  
  /**
   * Sets training size.
   *
   * @param trainingSize the training size
   * @return the training size
   */
  public HoldoverSampleTrainable setTrainingSize(final int trainingSize) {
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
    this.trainingSize += increaseMultiplier * Math.pow(this.trainingSize, increasePower);
    final Tensor[][] rawData = trainingData;
    assert 0 < rawData.length;
    assert 0 < getTrainingSize();
    Stream<Tensor[]> stream = Arrays.stream(rawData).parallel();
    if (shuffled) {
      stream = stream.sorted(Comparator.comparingLong(y -> System.identityHashCode(y) ^ this.hash));
    }
    this.sampledData = Stream.concat(
      stream.limit((long) Math.min(Math.max(getTrainingSize(), trainingSizeMin), trainingSizeMax) - holdoverData.length),
      Arrays.stream(holdoverData)
    ).toArray(i -> new Tensor[i][]);
  }
  
  /**
   * Sets training size.
   *
   * @param trainingSize the training size
   * @return the training size
   */
  public HoldoverSampleTrainable setTrainingSize(double trainingSize) {
    this.trainingSize = trainingSize;
    return this;
  }
  
  /**
   * Gets training size max.
   *
   * @return the training size max
   */
  public int getTrainingSizeMax() {
    return trainingSizeMax;
  }
  
  /**
   * Sets training size max.
   *
   * @param trainingSizeMax the training size max
   * @return the training size max
   */
  public HoldoverSampleTrainable setTrainingSizeMax(int trainingSizeMax) {
    this.trainingSizeMax = trainingSizeMax;
    return this;
  }
  
  /**
   * Gets training size min.
   *
   * @return the training size min
   */
  public int getTrainingSizeMin() {
    return trainingSizeMin;
  }
  
  /**
   * Sets training size min.
   *
   * @param trainingSizeMin the training size min
   * @return the training size min
   */
  public HoldoverSampleTrainable setTrainingSizeMin(int trainingSizeMin) {
    this.trainingSizeMin = trainingSizeMin;
    return this;
  }
  
  /**
   * Is shuffled boolean.
   *
   * @return the boolean
   */
  public boolean isShuffled() {
    return shuffled;
  }
  
  /**
   * Sets shuffled.
   *
   * @param shuffled the shuffled
   * @return the shuffled
   */
  public HoldoverSampleTrainable setShuffled(boolean shuffled) {
    this.shuffled = shuffled;
    return this;
  }
  
  /**
   * Gets increase multiplier.
   *
   * @return the increase multiplier
   */
  public double getIncreaseMultiplier() {
    return increaseMultiplier;
  }
  
  /**
   * Sets increase multiplier.
   *
   * @param increaseMultiplier the increase multiplier
   * @return the increase multiplier
   */
  public HoldoverSampleTrainable setIncreaseMultiplier(double increaseMultiplier) {
    this.increaseMultiplier = increaseMultiplier;
    return this;
  }
  
  /**
   * Gets increase power.
   *
   * @return the increase power
   */
  public double getIncreasePower() {
    return increasePower;
  }
  
  /**
   * Sets increase power.
   *
   * @param increasePower the increase power
   * @return the increase power
   */
  public HoldoverSampleTrainable setIncreasePower(double increasePower) {
    this.increasePower = increasePower;
    return this;
  }
  
  /**
   * Gets holdover fraction.
   *
   * @return the holdover fraction
   */
  public double getHoldoverFraction() {
    return holdoverFraction;
  }
  
  /**
   * Sets holdover fraction.
   *
   * @param holdoverFraction the holdover fraction
   * @return the holdover fraction
   */
  public HoldoverSampleTrainable setHoldoverFraction(double holdoverFraction) {
    this.holdoverFraction = holdoverFraction;
    return this;
  }
}
