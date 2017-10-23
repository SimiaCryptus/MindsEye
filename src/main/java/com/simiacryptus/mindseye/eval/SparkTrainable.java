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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.rdd.RDD;
import org.apache.spark.storage.StorageLevel;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * The type Spark trainable.
 */
public class SparkTrainable implements Trainable {
  
  protected final int sampleSize;
  protected int partitions;
  
  public StorageLevel getStorageLevel() {
    return storageLevel;
  }
  
  public CachedTrainable<SparkTrainable> cached() {
    return new CachedTrainable<SparkTrainable>(this);
  }
  
  public SparkTrainable setStorageLevel(StorageLevel storageLevel) {
    this.storageLevel = storageLevel;
    resetSampling();
    return this;
  }
  
  public int getPartitions() {
    return Math.max(1,partitions);
  }
  
  public SparkTrainable setPartitions(int partitions) {
    if(1 > partitions) throw new IllegalArgumentException();
    this.partitions = partitions;
    return this;
  }
  
  public boolean isVerbose() {
    return verbose;
  }
  
  public void setVerbose(boolean verbose) {
    this.verbose = verbose;
  }
  
  protected static class ReducableResult implements Serializable {
    /**
     * The Deltas.
     */
    public final Map<String, double[]> deltas;
    /**
     * The Sum.
     */
    public final double sum;
    
    /**
     * Instantiates a new Reducable result.
     *  @param deltas the deltas
     * @param sum    the sum
     */
    public ReducableResult(Map<String, double[]> deltas, double sum) {
      this.deltas = deltas;
      this.sum = sum;
    }
    
    /**
     * Accumulate.
     *
     * @param source the source
     */
    public void accumulate(DeltaSet source) {
      Map<String, NNLayer> idIndex = source.map.entrySet().stream().collect(Collectors.toMap(
        e -> e.getKey().id.toString(), e -> e.getKey()
      ));
      deltas.forEach((k, v) -> source.get(idIndex.get(k), (double[]) null).accumulate(v));
    }
    
    /**
     * Add spark trainable . reducable result.
     *
     * @param right the right
     * @return the spark trainable . reducable result
     */
    public SparkTrainable.ReducableResult add(SparkTrainable.ReducableResult right) {
      HashMap<String, double[]> map = new HashMap<>();
      Set<String> keys = Stream.concat(deltas.keySet().stream(), right.deltas.keySet().stream()).collect(Collectors.toSet());
      for (String key : keys) {
        double[] l = deltas.get(key);
        double[] r = right.deltas.get(key);
        if (null != r) {
          if (null != l) {
            assert (l.length == r.length);
            double[] x = new double[l.length];
            for (int i = 0; i < l.length; i++) x[i] = l[i] + r[i];
            map.put(key, x);
          }
          else {
            map.put(key, r);
          }
        }
        else {
          assert (null != l);
          map.put(key, l);
        }
      }
      return new SparkTrainable.ReducableResult(map, sum + right.sum);
    }
  
  }
  
  protected static void debug(String msg, Object... args) {
    String format = String.format(msg, args);
    System.out.println(format);
  }
  
  protected static class PartitionTask implements FlatMapFunction<Iterator<Tensor[]>, SparkTrainable.ReducableResult> {
    /**
     * The Network.
     */
    final NNLayer network;
    boolean verbose = true;
    
    protected PartitionTask(NNLayer network) {
      this.network = network;
    }
    
    @Override
    public Iterator<SparkTrainable.ReducableResult> call(Iterator<Tensor[]> partition) throws Exception {
      long startTime = System.nanoTime();
      GpuTrainable trainable = new GpuTrainable(network);
      Tensor[][] tensors = SparkTrainable.getStream(partition).toArray(i -> new Tensor[i][]);
      if(verbose) debug("Materialized %s records in %4f sec", tensors.length, (System.nanoTime() - startTime) * 1e-9);
      PointSample measure = trainable.setData(Arrays.asList(tensors)).measure();
      assert (measure != null);
      return Arrays.asList(SparkTrainable.getResult(measure.delta, new double[]{measure.value})).iterator();
    }
  }
  
  protected static SparkTrainable.ReducableResult getResult(DeltaSet delta, double[] values) {
    Map<String, double[]> deltas = delta.map.entrySet().stream().collect(Collectors.toMap(
      e -> e.getKey().id.toString(), e -> e.getValue().getDelta()
    ));
    return new SparkTrainable.ReducableResult(deltas, Arrays.stream(values).sum());
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
    TensorList resultData = result.getData();
    assert (resultData.stream().allMatch(x -> x.dim() == 1));
    assert (resultData.stream().allMatch(x -> Arrays.stream(x.getData()).allMatch(Double::isFinite)));
    double sum = resultData.stream().mapToDouble(x -> Arrays.stream(x.getData()).sum()).sum();
    return new PointSample(deltaSet, stateBackup, sum);
  }
  
  protected DeltaSet getDelta(SparkTrainable.ReducableResult reduce) {
    DeltaSet deltaSet = new DeltaSet();
    Tensor[] prototype = dataRDD.toJavaRDD().take(1).get(0);
    NNResult result = CudaExecutionContext.gpuContexts.map(exe->network.eval(exe, NNResult.batchResultArray(new Tensor[][]{prototype})));
    result.accumulate(deltaSet, 0);
    reduce.accumulate(deltaSet);
    return deltaSet;
  }
  
  protected final RDD<Tensor[]> dataRDD;
  protected RDD<Tensor[]> sampledRDD;
  protected final NNLayer network;
  protected boolean verbose = true;
  
  /**
   * Instantiates a new Spark trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   */
  public SparkTrainable(RDD<Tensor[]> trainingData, NNLayer network) {
    this(trainingData, network, -1);
  }
  
  public SparkTrainable(RDD<Tensor[]> trainingData, NNLayer network, int sampleSize) {
    this.dataRDD = trainingData;
    this.network = network;
    this.sampleSize = sampleSize;
    this.setPartitions(Math.max(1,dataRDD.sparkContext().executorEnvs().size()));
    resetSampling();
  }
  
  @Override
  public Trainable.PointSample measure() {
    long time1 = System.nanoTime();
    JavaRDD<ReducableResult> mapPartitions = this.sampledRDD.toJavaRDD().mapPartitions(new PartitionTask(network));
    long time2 = System.nanoTime();
    SparkTrainable.ReducableResult result = mapPartitions.reduce(SparkTrainable.ReducableResult::add);
    if(isVerbose()) System.out.println(String.format("Measure timing: %.3f / %.3f for %s items", (time2 - time1) * 1e-9, (System.nanoTime() - time2) * 1e-9, sampledRDD.count()));
    DeltaSet deltaSet = getDelta(result);
    DeltaSet stateSet = new DeltaSet();
    deltaSet.map.forEach((layer, layerDelta) -> {
      stateSet.get(layer, layerDelta.target).accumulate(layerDelta.target);
    });
    return new Trainable.PointSample(deltaSet, stateSet, result.sum);
  }
  
  @Override
  public boolean resetSampling() {
    assert (0 < sampleSize);
    long count = dataRDD.count();
    assert !this.dataRDD.isEmpty();
    if(null != this.sampledRDD) this.sampledRDD.unpersist(false);
    this.sampledRDD = dataRDD.sample(false, sampleSize * 1.0 / count, System.currentTimeMillis())
      .repartition(getPartitions(), null)
      .persist(getStorageLevel());
    assert !this.sampledRDD.isEmpty();
    System.out.println(String.format("Sampled %s items from main dataset of %s (%s) items", sampledRDD.count(), count, sampleSize));
    return true;
  }
  
  protected StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();
  
  @Override
  public void resetToFull() {
    this.sampledRDD = this.dataRDD.repartition(dataRDD.sparkContext().executorEnvs().size(), null).persist(getStorageLevel());
    System.out.println(String.format("Reset sample size to %s", sampledRDD.count()));
  }
  
  protected static Stream<Tensor[]> getStream(Iterator<Tensor[]> partition) {
    int characteristics = Spliterator.ORDERED;
    boolean parallel = false;
    Spliterator<Tensor[]> spliterator = Spliterators.spliteratorUnknownSize(partition, characteristics);
    return StreamSupport.stream(spliterator, parallel);
  }
  
  
}
