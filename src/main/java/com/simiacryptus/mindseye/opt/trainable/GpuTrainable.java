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

import com.simiacryptus.mindseye.layers.DeltaBuffer;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.util.ml.Tensor;

import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class GpuTrainable implements Trainable {
  protected final NNLayer network;
  protected int trainingSize = Integer.MAX_VALUE;
  protected boolean gcEachIteration = true;
  protected List<Tensor[]> sampledData;
  protected PointSample lastPtr;
  protected DeltaSet lastWeights;
  private boolean verbose = true;
  
  public GpuTrainable(NNLayer network) {
    this.network = network;
  }
  
  public GpuTrainable(NNLayer network, List<? extends Supplier<Tensor[]>> data) {
    this.network = network;
    this.setSampledData(data);
  }
  
  @Override
  public PointSample measure() {
    if(isStale(lastWeights)) {
      if(isVerbose()) System.out.println(String.format("Returning cached value"));
      return lastPtr;
    }
    PointSample result = GpuController.INSTANCE.distribute(sampledData,
      (list, dev) -> eval(NNResult.batchResultArray(list.stream().toArray(i1 -> new Tensor[i1][])), dev),
      (a, b) -> new PointSample(a.delta.add(b.delta), a.weights, a.value + b.value)
    );
    // Between each iteration is a great time to collect garbage, since the reachable object count will be at a low point.
    // Recommended JVM flags: -XX:+ExplicitGCInvokesConcurrent -XX:+UseConcMarkSweepGC
    if(gcEachIteration) GpuController.INSTANCE.cleanMemory();
    this.lastWeights = result.weights.copy();
    this.lastPtr = result;
    return result;
  }
  
  protected boolean isStale(DeltaSet weights) {
    if(null == weights) return false;
    List<DeltaBuffer> av = weights.vector();
    for(int i=0;i<av.size();i++) {
      DeltaBuffer ai = av.get(i);
      double[] aa = ai.getDelta();
      double[] bb = ai.target;
      if(aa[0] != bb[0]) return false;
      if(aa[aa.length/2-1] != bb[aa.length/2-1]) return false;
      if(aa[aa.length-1] != bb[aa.length-1]) return false;
    }
    return true;
  }
  
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
  
  public boolean isGcEachIteration() {
    return gcEachIteration;
  }
  
  public void setGcEachIteration(boolean gcEachIteration) {
    this.gcEachIteration = gcEachIteration;
  }
  
  protected void setSampledData(List<? extends Supplier<Tensor[]>> sampledData) {
    this.sampledData = sampledData.stream().parallel()
                         .map(x->x.get())
                         .collect(Collectors.toList());
  }
  
  public boolean isVerbose() {
    return verbose;
  }
  
  public void setVerbose(boolean verbose) {
    this.verbose = verbose;
  }
}
