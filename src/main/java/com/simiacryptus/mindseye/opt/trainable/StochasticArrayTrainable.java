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
import com.simiacryptus.mindseye.layers.DeltaBuffer;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Tensor;
import com.simiacryptus.util.ml.WeakCachedSupplier;

import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * The type Stochastic array trainable.
 *
 * TODO: Redesign this package. This class is absorbing too many features.
 */
public class StochasticArrayTrainable implements Trainable {
  
  private static final int LOW_MEM_USE = 4 * 1024 * 1024 * 1024;
  private final List<? extends Supplier<Tensor[]>> trainingData;
  private final NNLayer network;
  private long hash = Util.R.get().nextLong();
  private int trainingSize = Integer.MAX_VALUE;
  private boolean gcEachIteration = true;
  private List<Tensor[]> trainingNNResultArrays;
  private List<? extends Supplier<Tensor[]>> sampledData;
  private PointSample lastPtr;
  private DeltaSet lastWeights;
  private boolean verbose = true;
  
  /**
   * Instantiates a new Stochastic array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param trainingSize the training size
   */
  public StochasticArrayTrainable(Tensor[][] trainingData, NNLayer network, int trainingSize) {
    this(trainingData, network, trainingSize, trainingSize / CudaExecutionContext.gpuContexts.size());
  }
  
  /**
   * Instantiates a new Stochastic array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param trainingSize the training size
   * @param batchSize    the batch size
   */
  public StochasticArrayTrainable(Tensor[][] trainingData, NNLayer network, int trainingSize, int batchSize) {
    if(0 == trainingData.length) throw new IllegalArgumentException();
    this.trainingData = Arrays.stream(trainingData).map(obj->new WeakCachedSupplier<Tensor[]>(()->obj)).collect(Collectors.toList());
    this.network = network;
    this.trainingSize = trainingSize;
    resetSampling();
  }
  
  public StochasticArrayTrainable(List<? extends Supplier<Tensor[]>> trainingData, NNLayer network, int trainingSize) {
    this(trainingData, network, trainingSize, trainingSize);
  }
  
  public StochasticArrayTrainable(List<? extends Supplier<Tensor[]>> trainingData, NNLayer network, int trainingSize, int batchSize) {
    if(0 == trainingData.size()) throw new IllegalArgumentException();
    this.trainingData = trainingData;
    this.network = network;
    this.trainingSize = trainingSize;
    resetSampling();
  }
  
  Map<String, Double> deviceWeight = new HashMap<>();
  Map<String, Integer> deviceBatchSizes = new HashMap<>();
 
  @Override
  public PointSample measure() {
    if(isStale(lastWeights)) {
      if(verbose) System.out.println(String.format("Returning cached value"));
      return lastPtr;
    }
    List<CudaExecutionContext> devices = CudaExecutionContext.gpuContexts.getAll();
    Map<CudaExecutionContext, List<Tensor[]>> tasks = new HashMap<>();
    double weightSum = devices.stream().mapToDouble(d -> deviceWeight.getOrDefault(d.toString(), 1.0)).sum();
    int start = 0;
    for(int i=0;i<devices.size();i++) {
      CudaExecutionContext dev = devices.get(i);
      int sampleSize = (int) ((trainingNNResultArrays.size() / weightSum) * deviceWeight.getOrDefault(dev.toString(), 1.0));
      int end = start + sampleSize;
      tasks.put(dev, trainingNNResultArrays.subList(start, end));
      start = end;
    }
    List<PointSample> results = new ArrayList<>();
    List<Thread> threads = new ArrayList<>();
    for(Map.Entry<CudaExecutionContext, List<Tensor[]>> e : tasks.entrySet()) {
      Thread thread = new Thread(new Runnable() {
        @Override
        public void run() {
          List<Tensor[]> data = e.getValue();
          CudaExecutionContext gpu = e.getKey();
          Integer batchSize = deviceBatchSizes.getOrDefault(gpu.toString(), data.size());
          try {
            results.add(evaluate(data, gpu, batchSize));
          } catch (Throwable t) {
            if(isOom(t) && batchSize > 1) {
              batchSize = batchSize / 2;
              deviceBatchSizes.put(gpu.toString(), batchSize);
              cleanMemory();
              results.add(evaluate(data, gpu, batchSize)); // Second Chance
            } else {
              RuntimeException runtimeException = new RuntimeException("Failed executing " + this, t);
              runtimeException.printStackTrace(System.err);
              throw runtimeException;
            }
          }
        }
      });
      thread.start();
      threads.add(thread);
    }
    try {
      for(Thread thread : threads) thread.join();
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
    PointSample result = results.stream().reduce((a, b) -> new PointSample(a.delta.add(b.delta), a.weights, a.value + b.value)).get();
    // Between each iteration is a great time to collect garbage, since the reachable object count will be at a low point.
    // Recommended JVM flags: -XX:+ExplicitGCInvokesConcurrent -XX:+UseConcMarkSweepGC
    if(gcEachIteration) cleanMemory();
    this.lastWeights = result.weights.copy();
    this.lastPtr = result;
    return result;
  }
  
  private boolean isStale(DeltaSet weights) {
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
  
  private PointSample evaluate(List<Tensor[]> data, CudaExecutionContext gpu, int batchSize ) {
    long startNanos = System.nanoTime();
    PointSample deviceResult = Lists.partition(data, batchSize).stream().map(list -> {
      return eval(NNResult.batchResultArray(list.stream().toArray(i -> new Tensor[i][])), gpu);
    }).reduce((a, b) -> new PointSample(a.delta.add(b.delta), a.weights, a.value + b.value)).get();
    double time = (System.nanoTime() - startNanos) * 1.0 / 1e9;
    if(verbose) System.out.println(String.format("Device %s completed %s items in %s sec", gpu, data.size(), time));
    deviceWeight.put(gpu.toString(), data.size() / time);
    return deviceResult;
  }
  
  public void cleanMemory() {
    Tensor.clear();
    System.gc();
    System.runFinalization();
  }
  
  private static boolean isOom(Throwable t) {
    if(t instanceof OutOfMemoryError) return true;
    if(null != t.getCause() && t != t.getCause()) return isOom(t.getCause());
    return false;
  }
  
  private PointSample eval(NNResult[] input, CudaExecutionContext nncontext) {
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
  
  @Override
  public void resetToFull() {
    this.setSampledData(trainingData.stream().collect(Collectors.toList()));
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
  public int getTrainingSize() {
    return this.trainingSize;
  }
  
  /**
   * Sets training size.
   *
   * @param trainingSize the training size
   * @return the training size
   */
  public StochasticArrayTrainable setTrainingSize(final int trainingSize) {
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
    lastPtr = null;
    lastWeights = null;
    assert 0 < trainingData.size();
    assert 0 < getTrainingSize();
    this.setSampledData(trainingData.stream().parallel() //
                           .sorted(Comparator.comparingLong(y -> System.identityHashCode(y) ^ this.hash)) //
                           .filter(x->x.get()!=null)
                           .limit(getTrainingSize()) //
                           .collect(Collectors.toList()));
  }
  
  public boolean isGcEachIteration() {
    return gcEachIteration;
  }
  
  public void setGcEachIteration(boolean gcEachIteration) {
    this.gcEachIteration = gcEachIteration;
  }
  
  protected void setSampledData(List<? extends Supplier<Tensor[]>> sampledData) {
    this.sampledData = sampledData;
    //Lists.partition(, batchSize)
    //.map(xx -> NNResult.batchResultArray(xx.stream().parallel().map(x->x.get()).toArray(i->new Tensor[i][])))
    this.trainingNNResultArrays = this.sampledData.stream().parallel()
                                    .map(x->x.get())
                                    .collect(Collectors.toList());
  }
  
}
