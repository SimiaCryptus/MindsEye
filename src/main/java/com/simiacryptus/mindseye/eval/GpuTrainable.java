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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.cudnn.*;
import jcuda.runtime.JCuda;

import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * The type Gpu trainable.
 */
public class GpuTrainable implements DataTrainable {
  
  /**
   * The Network.
   */
  protected final NNLayer network;
  /**
   * The Gc each iteration.
   */
  protected boolean gcEachIteration = true;
  /**
   * The Sampled data.
   */
  protected List<Tensor[]> data;
  
  private boolean verbose = false;
  
  public GpuTrainable(NNLayer network) {
    this.network = network;
    this.data = null;
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
        (list, dev) -> eval(list, dev),
        (a, b) -> a.add(b)
      );
      //          System.out.println(String.format("Evaluated to %s delta arrays", deltaSet.map.size()));
  
      assert (null != result);
      // Between each iteration is a great time to collect garbage, since the reachable object count will be at a low point.
      // Recommended JVM flags: -XX:+ExplicitGCInvokesConcurrent -XX:+UseConcMarkSweepGC
      if (gcEachIteration) GpuController.INSTANCE.cleanMemory();
      return result;
    } catch (Exception e) {
      if(retries > 0) {
        GpuController.INSTANCE.cleanMemory();
        synchronized (CudaResource.gpuGeneration) {
          for(Map.Entry<CudaExecutionContext, ExecutorService> entry : GpuController.INSTANCE.getGpuDriverThreads().asMap().entrySet()) {
            CudaResource.gpuGeneration.incrementAndGet();
            try {
              entry.getValue().submit(()->{
                CudaPtr.getGpuStats(entry.getKey().getDeviceNumber()).usedMemory.set(0);
                return JCuda.cudaDeviceReset();
              }).get();
            } catch (InterruptedException e1) {
              throw new GpuError(e1);
            } catch (ExecutionException e1) {
              throw new GpuError(e1);
            }
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
   * @param list the input
   * @param nncontext the nncontext
   * @return the point sample
   */
  protected PointSample eval(List<Tensor[]> list, CudaExecutionContext nncontext) {
    NNResult result = network.eval(nncontext, getNNContext(list));
    TensorList resultData = result.getData();
    assert (resultData.stream().allMatch(x -> x.dim() == 1));
    assert (resultData.stream().allMatch(x -> Arrays.stream(x.getData()).allMatch(Double::isFinite)));
    DoubleStream stream = resultData.stream().flatMapToDouble(x -> Arrays.stream(x.getData()));
    DoubleSummaryStatistics statistics = stream.summaryStatistics();
    double sum = statistics.getAverage();
    DeltaSet deltaSet = new DeltaSet();
    result.accumulate(deltaSet, 1.0 / statistics.getCount());
    //System.out.println(String.format("Evaluated to %s delta arrays", deltaSet.map.size()));
    assert (deltaSet.vector().stream().allMatch(x -> Arrays.stream(x.getDelta()).allMatch(Double::isFinite)));
    DeltaSet stateBackup = new DeltaSet();
    deltaSet.map.forEach((layer, layerDelta) -> {
      stateBackup.get(layer, layerDelta.target).accumulate(layerDelta.target);
    });
    assert (stateBackup.vector().stream().allMatch(x -> Arrays.stream(x.getDelta()).allMatch(Double::isFinite)));
    return new PointSample(deltaSet, stateBackup, sum, statistics.getCount());
  }
  
  private NNResult[] getNNContext(List<Tensor[]> list) {
    Tensor[] head = list.get(0);
    return IntStream.range(0, head.length).mapToObj(inputIndex -> {
      Tensor[] array = IntStream.range(0, list.size()).mapToObj(trainingExampleId -> {
          return list.get(trainingExampleId)[inputIndex];
        }
      ).toArray(i -> new Tensor[i]);
      return new ConstNNResult(array);
    }).toArray(x1 -> new NNResult[x1]);
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
  public Trainable setData(List<Tensor[]> sampledData) {
    assert !sampledData.isEmpty();
    this.data = sampledData;
    return this;
  }
  
  @Override
  public Tensor[][] getData() {
    return data.toArray(new Tensor[][]{});
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
  public GpuTrainable setVerbose(boolean verbose) {
    this.verbose = verbose;
    return this;
  }
  
  private static class InputNNLayer extends NNLayer {
    @Override
    public NNResult eval(NNExecutionContext nncontext, NNResult[] array) {
      throw new IllegalStateException();
    }
    
    @Override
    public JsonObject getJson() {
      throw new IllegalStateException();
    }
    
    @Override
    public List<double[]> state() {
      throw new IllegalStateException();
    }
  }
}