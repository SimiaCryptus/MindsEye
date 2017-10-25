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

import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.cudnn.CuDNN;
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import com.simiacryptus.mindseye.layers.cudnn.CudaPtr;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import jcuda.runtime.JCuda;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

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
  protected List<Tensor[]> data;
  
  private boolean verbose = false;
  
  /**
   * Instantiates a new Gpu trainable.
   *
   * @param network the network
   */
  public GpuTrainable(NNLayer network) {
    this.network = network;
  }
  
  @Override
  public PointSample measure() {
    return measure(3);
  }
  
  /**
   * Measure point sample.
   *
   * @param retries the retries
   * @return the point sample
   */
  public PointSample measure(int retries) {
    try {
      assert !data.isEmpty();
      PointSample result = GpuController.INSTANCE.distribute(data,
        (list, dev) -> eval(NNResult.batchResultArray(list.stream().toArray(i1 -> new Tensor[i1][])), dev),
        (a, b) -> a.add(b)
      );
      assert (null != result);
      // Between each iteration is a great time to collect garbage, since the reachable object count will be at a low point.
      // Recommended JVM flags: -XX:+ExplicitGCInvokesConcurrent -XX:+UseConcMarkSweepGC
      if (gcEachIteration) GpuController.INSTANCE.cleanMemory();
      return result;
    } catch (Exception e) {
      if(retries > 0) {
        GpuController.INSTANCE.cleanMemory();
        for(Map.Entry<CuDNN, ExecutorService> entry : GpuController.INSTANCE.getGpuDriverThreads().asMap().entrySet()) {
          try {
            entry.getValue().submit(()->JCuda.cudaDeviceReset()).get();
          } catch (InterruptedException e1) {
            throw new GpuError(e1);
          } catch (ExecutionException e1) {
            throw new GpuError(e1);
          }
        }
        CudaPtr.METRICS.invalidateAll();
        GpuController.INSTANCE.cleanMemory();
        return measure(retries-1);
      } else {
        throw e;
      }
    }
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
    TensorList resultData = result.getData();
    assert (resultData.stream().allMatch(x -> x.dim() == 1));
    assert (resultData.stream().allMatch(x -> Arrays.stream(x.getData()).allMatch(Double::isFinite)));
    DoubleStream stream = resultData.stream().flatMapToDouble(x -> Arrays.stream(x.getData()));
    DoubleSummaryStatistics statistics = stream.summaryStatistics();
    double sum = statistics.getAverage();
    DeltaSet deltaSet = new DeltaSet();
    result.accumulate(deltaSet, 1.0 / statistics.getCount());
    assert (deltaSet.vector().stream().allMatch(x -> Arrays.stream(x.getDelta()).allMatch(Double::isFinite)));
    DeltaSet stateBackup = new DeltaSet();
    deltaSet.map.forEach((layer, layerDelta) -> {
      stateBackup.get(layer, layerDelta.target).accumulate(layerDelta.target);
    });
    assert (stateBackup.vector().stream().allMatch(x -> Arrays.stream(x.getDelta()).allMatch(Double::isFinite)));

    return new PointSample(deltaSet, stateBackup, sum, statistics.getCount());
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
   * @param data the sampled data
   */
  protected void setDataSupplier(List<? extends Supplier<Tensor[]>> data) {
    setData(data.stream().parallel().map(x -> x.get()).collect(Collectors.toList()));
  }
  
  /**
   * Sets data.
   *
   * @param sampledData the sampled data
   * @return the data
   */
  public GpuTrainable setData(List<Tensor[]> sampledData) {
    assert !sampledData.isEmpty();
    this.data = sampledData;
    return this;
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
