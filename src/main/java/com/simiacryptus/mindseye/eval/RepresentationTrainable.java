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

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * The type Gpu trainable.
 */
public class RepresentationTrainable implements Trainable {
  
  /**
   * The Network.
   */
  protected final NNLayer network;
  private final NNResult[] dataContext;
  private final Tensor[][] data;
  protected boolean gcEachIteration = true;
  private boolean verbose = false;
  private boolean[] dynamicMask = null;
  
  public RepresentationTrainable(NNLayer network, Tensor[][] data, boolean[] dynamicMask) {
    assert 0<data.length;
    this.network = network;
    this.dynamicMask = dynamicMask;
    this.data = data;
    int cols = data[0].length;
    int rows = data.length;
    this.dataContext = IntStream.range(0, cols).mapToObj(col -> {
      Tensor[] tensors = IntStream.range(0, rows).mapToObj(row -> data[row][col]).toArray(i -> new Tensor[i]);
      if (null == dynamicMask || !dynamicMask[col]) {
        return new ConstNNResult(tensors);
      }
      else {
        return new NNResult(tensors) {
          PlaceholderLayer[] layer = IntStream.range(0,tensors.length)
                                       .mapToObj(i->new PlaceholderLayer())
                                       .toArray(i->new PlaceholderLayer[i]);
        
          @Override
          public void accumulate(DeltaSet buffer, TensorList delta) {
            //System.out.println("Accumulating data");
            for (int index = 0; index < delta.length(); index++) {
              double[] doubles = delta.get(index).getData();
              //System.out.println(String.format("Accumulating data[%s] => %s", index, Long.toHexString(System.identityHashCode(doubles))));
              Delta deltaObj = buffer.get(layer[index], tensors[index]);
              deltaObj.accumulate(doubles);
            }
          }
        
          @Override
          public boolean isAlive() {
            return true;
          }
        };
      }
    }).toArray(x1 -> new NNResult[x1]);
  }
  
  @Override
  public PointSample measure(boolean isStatic) {
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
      CudaExecutionContext executionContext = CudaExecutionContext.gpuContexts.getAll().get(0);
      ExecutorService executorService = GpuController.INSTANCE.getGpuDriverThreads().get(executionContext);
      PointSample result = executorService.submit(()->eval(executionContext)).get();
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
              ExecutorService executorService = entry.getValue();
              CudaExecutionContext executionContext = entry.getKey();
              executorService.submit(()->{
                CudaPtr.getGpuStats(executionContext.getDeviceNumber()).usedMemory.set(0);
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
        throw new RuntimeException(e);
      }
    }
  }
  
  /**
   * Eval point sample.
   *
   * @param nncontext the nncontext
   * @return the point sample
   */
  protected PointSample eval(CudaExecutionContext nncontext) {
    NNResult result = network.eval(nncontext, dataContext);
    TensorList resultData = result.getData();
    assert (resultData.stream().allMatch(x -> x.dim() == 1));
    assert (resultData.stream().allMatch(x -> Arrays.stream(x.getData()).allMatch(Double::isFinite)));
    DoubleStream stream = resultData.stream().flatMapToDouble(x -> Arrays.stream(x.getData()));
    DoubleSummaryStatistics statistics = stream.summaryStatistics();
    double sum = statistics.getAverage();
    DeltaSet deltaSet = new DeltaSet();
    result.accumulate(deltaSet, 1.0 / statistics.getCount());
    //System.out.println(String.format("Evaluated to %s delta arrays", deltaSet.run.size()));
    assert (deltaSet.vector().stream().allMatch(x -> Arrays.stream(x.getDelta()).allMatch(Double::isFinite)));
    DeltaSet stateBackup = new DeltaSet();
    deltaSet.map.forEach((layer, layerDelta) -> {
      stateBackup.get(layer, layerDelta.target).accumulate(layerDelta.target);
    });
    assert (stateBackup.vector().stream().allMatch(x -> Arrays.stream(x.getDelta()).allMatch(Double::isFinite)));
    return new PointSample(deltaSet, stateBackup, sum, statistics.getCount());
  }
  
  public Tensor[][] getData() {
    return data;
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
  public RepresentationTrainable setVerbose(boolean verbose) {
    this.verbose = verbose;
    return this;
  }
  
  public boolean[] getDynamicMask() {
    return dynamicMask;
  }
  
  public RepresentationTrainable setDynamicMask(boolean[] dynamicMask) {
    this.dynamicMask = dynamicMask;
    return this;
  }
  
  private static class PlaceholderLayer extends NNLayer {
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
