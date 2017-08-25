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
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.util.ml.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * The type Gpu trainable.
 */
public class GpuTrainable implements Trainable {
  /**
   * The Network.
   */
  protected final NNLayer network;
  /**
   * The Training size.
   */
  protected int trainingSize = Integer.MAX_VALUE;
  /**
   * The Gc each iteration.
   */
  protected boolean gcEachIteration = true;
  /**
   * The Sampled data.
   */
  protected List<Tensor[]> sampledData;
  /**
   * The Last ptr.
   */
  protected PointSample lastPtr;
  /**
   * The Last weights.
   */
  protected DeltaSet lastWeights;
  private boolean verbose = true;
  
  /**
   * Instantiates a new Gpu trainable.
   *
   * @param network the network
   */
  public GpuTrainable(NNLayer network) {
    this.network = network;
  }
  
  /**
   * Instantiates a new Gpu trainable.
   *
   * @param network the network
   * @param data    the data
   */
  public GpuTrainable(NNLayer network, List<? extends Supplier<Tensor[]>> data) {
    this.network = network;
    this.setSampledData(data);
  }
  
  @Override
  public PointSample measure() {
    if(null != lastWeights && !lastWeights.isDifferent()) {
      if(isVerbose()) System.out.println(String.format("Returning cached value"));
      return lastPtr;
    }
    PointSample result = GpuController.INSTANCE.distribute(sampledData,
      (list, dev) -> eval(NNResult.batchResultArray(list.stream().toArray(i1 -> new Tensor[i1][])), dev),
      (a, b) -> a.add(b)
    );
    // Between each iteration is a great time to collect garbage, since the reachable object count will be at a low point.
    // Recommended JVM flags: -XX:+ExplicitGCInvokesConcurrent -XX:+UseConcMarkSweepGC
    if(gcEachIteration) GpuController.INSTANCE.cleanMemory();
    this.lastWeights = result.weights.copy();
    this.lastPtr = result;
    return result;
  }
  
  /**
   * Eval point sample.
   *
   * @param input     the input
   * @param nncontext the nncontext
   * @return the point sample
   */
  protected PointSample eval(NNResult[] input, CudaExecutionContext nncontext) {
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
  }
  
  /**
   * Is gc each iteration boolean.
   *
   * @return the boolean
   */
  public boolean isGcEachIteration() {
    return gcEachIteration;
  }
  
  /**
   * Sets gc each iteration.
   *
   * @param gcEachIteration the gc each iteration
   */
  public void setGcEachIteration(boolean gcEachIteration) {
    this.gcEachIteration = gcEachIteration;
  }
  
  /**
   * Sets sampled data.
   *
   * @param sampledData the sampled data
   */
  protected void setSampledData(List<? extends Supplier<Tensor[]>> sampledData) {
    this.sampledData = sampledData.stream().parallel()
                         .map(x->x.get())
                         .collect(Collectors.toList());
  }
  
  /**
   * Is verbose boolean.
   *
   * @return the boolean
   */
  public boolean isVerbose() {
    return verbose;
  }
  
  /**
   * Sets verbose.
   *
   * @param verbose the verbose
   */
  public void setVerbose(boolean verbose) {
    this.verbose = verbose;
  }
}
