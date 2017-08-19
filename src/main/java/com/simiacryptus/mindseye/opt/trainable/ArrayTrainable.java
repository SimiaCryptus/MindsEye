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
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import com.simiacryptus.util.ml.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

/**
 * The type Array trainable.
 */
public class ArrayTrainable implements Trainable {
  
  private final Tensor[][] trainingData;
  private final NNLayer network;
  private final int batchSize;
  private boolean parallel = true;
  
  /**
   * Instantiates a new Array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   */
  public ArrayTrainable(Tensor[][] trainingData, NNLayer network) {
    this(trainingData, network, trainingData.length);
  }
  
  /**
   * Instantiates a new Array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param batchSize    the batch size
   */
  public ArrayTrainable(Tensor[][] trainingData, NNLayer network, int batchSize) {
    if(0 == trainingData.length) throw new IllegalArgumentException();
    this.trainingData = trainingData;
    this.network = network;
    this.batchSize = batchSize;
    resetSampling();
  }
  
  @Override
  public PointSample measure() {
    List<List<Tensor[]>> collection = batchSize<trainingData.length?
                                        Lists.partition(Arrays.asList(trainingData), batchSize)
                                        :Arrays.asList(Arrays.asList(trainingData));
    Stream<List<Tensor[]>> stream = collection.stream();
    if(isParallel()) stream = stream.parallel();
    return stream.map(trainingData->{
      NNResult[] input = NNResult.batchResultArray(trainingData.toArray(new Tensor[][]{}));
      NNResult result = CudaExecutionContext.gpuContexts.map(ctx->network.eval(ctx, input));
      DeltaSet deltaSet = new DeltaSet();
      CudaExecutionContext.gpuContexts.apply(ctx->result.accumulate(deltaSet));
      DeltaSet stateSet = new DeltaSet();
      deltaSet.map.forEach((layer, layerDelta) -> {
        stateSet.get(layer, layerDelta.target).accumulate(layerDelta.target);
      });
      assert (result.getData().stream().allMatch(x -> x.dim() == 1));
      double sum = result.getData().stream().mapToDouble(x -> x.getData()[0]).sum();
      return new PointSample(deltaSet.scale(1.0/trainingData.size()), stateSet, sum / trainingData.size());
    }).reduce((a,b)->new PointSample(a.delta.add(b.delta),a.weights, a.value + b.value)).get();
  }
  
  @Override
  public void resetToFull() {
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
   * Is parallel boolean.
   *
   * @return the boolean
   */
  public boolean isParallel() {
    return parallel;
  }
  
  /**
   * Sets parallel.
   *
   * @param parallel the parallel
   * @return the parallel
   */
  public ArrayTrainable setParallel(boolean parallel) {
    this.parallel = parallel;
    return this;
  }
}
