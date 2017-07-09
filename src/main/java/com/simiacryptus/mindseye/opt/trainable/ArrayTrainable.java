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
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.ml.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class ArrayTrainable implements Trainable {
  
  private final Tensor[][] trainingData;
  private final DAGNetwork network;
  private final int batchSize;
  private boolean parallel = false;
  
  public ArrayTrainable(Tensor[][] trainingData, DAGNetwork network) {
    this(trainingData, network, trainingData.length);
  }
  public ArrayTrainable(Tensor[][] trainingData, DAGNetwork network, int batchSize) {
    this.trainingData = trainingData;
    this.network = network;
    this.batchSize = batchSize;
    resetSampling();
  }
  
  @Override
  public PointSample measure() {
    Stream<List<Tensor[]>> stream = Lists.partition(Arrays.asList(trainingData), batchSize).stream();
    if(isParallel()) stream = stream.parallel();
    return stream.map(trainingData->{
      NNResult[] input = NNResult.batchResultArray(trainingData.toArray(new Tensor[][]{}));
      NNResult result = network.eval(input);
      DeltaSet deltaSet = new DeltaSet();
      result.accumulate(deltaSet);
      DeltaSet stateSet = new DeltaSet();
      deltaSet.map.forEach((layer, layerDelta) -> {
        stateSet.get(layer, layerDelta.target).accumulate(layerDelta.target);
      });
      assert (Arrays.stream(result.data).allMatch(x -> x.dim() == 1));
      double meanValue = Arrays.stream(result.data).mapToDouble(x -> x.getData()[0]).average().getAsDouble();
      return new PointSample(deltaSet, stateSet, meanValue);
    }).reduce((a,b)->new PointSample(a.weights, a.delta.add(b.delta),a.value + b.value)).get();
  }
  
  @Override
  public void resetToFull() {
  }
  
  public int getBatchSize() {
    return batchSize;
  }
  
  public boolean isParallel() {
    return parallel;
  }
  
  public ArrayTrainable setParallel(boolean parallel) {
    this.parallel = parallel;
    return this;
  }
}
