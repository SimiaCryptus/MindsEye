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
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
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
 * A training implementation which holds data as a Spark RDD 
 * and distributes network evaluation over the partitions.
 */
public class SparkTrainable implements Trainable {
  
  /**
   * The Sample size.
   */
  protected final int sampleSize;
  /**
   * The Data rdd.
   */
  protected final RDD<Tensor[]> dataRDD;
  /**
   * The Network.
   */
  protected final NNLayer network;
  /**
   * The Partitions.
   */
  protected int partitions;
  /**
   * The Sampled rdd.
   */
  protected RDD<Tensor[]> sampledRDD;
  /**
   * The Verbose.
   */
  protected boolean verbose = true;
  /**
   * The Storage level.
   */
  protected StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();
  private long seed;
  
  /**
   * Instantiates a new Spark trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   */
  public SparkTrainable(RDD<Tensor[]> trainingData, NNLayer network) {
    this(trainingData, network, -1);
  }
  
  /**
   * Instantiates a new Spark trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param sampleSize   the sample size
   */
  public SparkTrainable(RDD<Tensor[]> trainingData, NNLayer network, int sampleSize) {
    this.dataRDD = trainingData;
    this.network = network;
    this.sampleSize = sampleSize;
    this.setPartitions(Math.max(1, dataRDD.sparkContext().executorEnvs().size()));
    reseed(seed);
  }
  
  /**
   * Debug.
   *
   * @param msg  the msg
   * @param args the args
   */
  protected static void debug(String msg, Object... args) {
    String format = String.format(msg, args);
    System.out.println(format);
  }
  
  /**
   * Gets result.
   *
   * @param delta  the delta
   * @param values the values
   * @return the result
   */
  protected static SparkTrainable.ReducableResult getResult(DeltaSet<NNLayer> delta, double[] values) {
    Map<String, double[]> deltas = delta.getMap().entrySet().stream().collect(Collectors.toMap(
      e -> e.getKey().getId(), e -> e.getValue().getDelta()
    ));
    return new SparkTrainable.ReducableResult(deltas, values.length, Arrays.stream(values).sum());
  }
  
  /**
   * Gets stream.
   *
   * @param partition the partition
   * @return the stream
   */
  protected static Stream<Tensor[]> getStream(Iterator<Tensor[]> partition) {
    int characteristics = Spliterator.ORDERED;
    boolean parallel = false;
    Spliterator<Tensor[]> spliterator = Spliterators.spliteratorUnknownSize(partition, characteristics);
    return StreamSupport.stream(spliterator, parallel);
  }
  
  /**
   * Gets storage level.
   *
   * @return the storage level
   */
  public StorageLevel getStorageLevel() {
    return storageLevel;
  }
  
  /**
   * Sets storage level.
   *
   * @param storageLevel the storage level
   * @return the storage level
   */
  public SparkTrainable setStorageLevel(StorageLevel storageLevel) {
    this.storageLevel = storageLevel;
    reseed(seed);
    return this;
  }
  
  /**
   * Gets partitions.
   *
   * @return the partitions
   */
  public int getPartitions() {
    return Math.max(1, partitions);
  }
  
  /**
   * Sets partitions.
   *
   * @param partitions the partitions
   * @return the partitions
   */
  public SparkTrainable setPartitions(int partitions) {
    if (1 > partitions) throw new IllegalArgumentException();
    this.partitions = partitions;
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
  
  /**
   * Gets delta.
   *
   * @param reduce the reduce
   * @return the delta
   */
  protected DeltaSet getDelta(SparkTrainable.ReducableResult reduce) {
    DeltaSet deltaSet = new DeltaSet();
    Tensor[] prototype = dataRDD.toJavaRDD().take(1).get(0);
    NNResult result = GpuController.call(exe -> network.eval(exe, NNResult.batchResultArray(new Tensor[][]{prototype})));
    result.accumulate(deltaSet, 0);
    reduce.accumulate(deltaSet);
    return deltaSet;
  }
  
  @Override
  public PointSample measure(boolean isStatic, TrainingMonitor monitor) {
    long time1 = System.nanoTime();
    JavaRDD<ReducableResult> mapPartitions = this.sampledRDD.toJavaRDD().mapPartitions(new PartitionTask(network));
    long time2 = System.nanoTime();
    SparkTrainable.ReducableResult result = mapPartitions.reduce(SparkTrainable.ReducableResult::add);
    if (isVerbose()) {
      System.out.println(String.format("Measure timing: %.3f / %.3f for %s items", (time2 - time1) * 1e-9, (System.nanoTime() - time2) * 1e-9, sampledRDD.count()));
    }
    DeltaSet deltaSet = getDelta(result);
    return new PointSample(deltaSet, new StateSet(deltaSet), result.sum, 0.0, result.count).normalize();
  }
  
  @Override
  public boolean reseed(long seed) {
    this.seed = seed;
    assert (0 < sampleSize);
    long count = dataRDD.count();
    assert !this.dataRDD.isEmpty();
    if (null != this.sampledRDD) this.sampledRDD.unpersist(false);
    this.sampledRDD = dataRDD.sample(false, sampleSize * 1.0 / count, System.currentTimeMillis())
      .repartition(getPartitions(), null)
      .persist(getStorageLevel());
    assert !this.sampledRDD.isEmpty();
    System.out.println(String.format("Sampled %s items from main dataset of %s (%s) items", sampledRDD.count(), count, sampleSize));
    return true;
  }
  
  /**
   * The type Reducable result.
   */
  protected static class ReducableResult implements Serializable {
    /**
     * The Deltas.
     */
    public final Map<String, double[]> deltas;
    /**
     * The Sum.
     */
    public final double sum;
    public final int count;
  
    /**
     * Instantiates a new Reducable result.
     *
     * @param deltas the deltas
     * @param sum    the sum
     */
    public ReducableResult(Map<String, double[]> deltas, int count, double sum) {
      this.deltas = deltas;
      this.count = count;
      this.sum = sum;
    }
  
    /**
     * Accumulate.
     *
     * @param source the source
     */
    public void accumulate(DeltaSet<NNLayer> source) {
      Map<String, NNLayer> idIndex = source.getMap().entrySet().stream().collect(Collectors.toMap(
        e -> e.getKey().getId(), e -> e.getKey()
      ));
      deltas.forEach((k, v) -> source.get(idIndex.get(k), (double[]) null).addInPlace(v));
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
      return new SparkTrainable.ReducableResult(map, count+right.count, sum + right.sum);
    }
    
  }
  
  /**
   * The type Partition task.
   */
  protected static class PartitionTask implements FlatMapFunction<Iterator<Tensor[]>, SparkTrainable.ReducableResult> {
    /**
     * The Network.
     */
    final NNLayer network;
    /**
     * The Verbose.
     */
    boolean verbose = true;
  
    /**
     * Instantiates a new Partition task.
     *
     * @param network the network
     */
    protected PartitionTask(NNLayer network) {
      this.network = network;
    }
    
    @Override
    public Iterator<SparkTrainable.ReducableResult> call(Iterator<Tensor[]> partition) throws Exception {
      long startTime = System.nanoTime();
      GpuTrainable trainable = new GpuTrainable(network);
      Tensor[][] tensors = SparkTrainable.getStream(partition).toArray(i -> new Tensor[i][]);
      if (verbose) debug("Materialized %s records in %4f sec", tensors.length, (System.nanoTime() - startTime) * 1e-9);
      PointSample measure = trainable.setData(Arrays.asList(tensors)).measure(false, new TrainingMonitor() {
        @Override
        public void log(String msg) {
          debug(msg);
        }
      });
      assert (measure != null);
      return Arrays.asList(SparkTrainable.getResult(measure.delta, new double[]{measure.sum})).iterator();
    }
  }
  
  
}
