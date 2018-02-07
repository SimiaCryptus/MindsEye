/*
 * Copyright (c) 2018 by Andrew Charneski.
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
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.rdd.RDD;
import org.apache.spark.storage.StorageLevel;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * A training implementation which holds data as a Spark RDD and distributes network evaluation over the partitions.
 */
public class SparkTrainable extends TrainableBase {
  /**
   * The Logger.
   */
  static final Logger log = LoggerFactory.getLogger(SparkTrainable.class);
  
  /**
   * The Data rdd.
   */
  protected final RDD<Tensor[]> dataRDD;
  /**
   * The Network.
   */
  protected final NNLayer network;
  /**
   * The Sample size.
   */
  protected final int sampleSize;
  /**
   * The Partitions.
   */
  protected int partitions;
  /**
   * The Sampled rdd.
   */
  protected RDD<Tensor[]> sampledRDD;
  /**
   * The Storage level.
   */
  protected StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();
  /**
   * The Verbose.
   */
  protected boolean verbose = true;
  private long seed;
  
  /**
   * Instantiates a new Spark trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   */
  public SparkTrainable(final RDD<Tensor[]> trainingData, final NNLayer network) {
    this(trainingData, network, -1);
  }
  
  /**
   * Instantiates a new Spark trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param sampleSize   the sample size
   */
  public SparkTrainable(final RDD<Tensor[]> trainingData, final NNLayer network, final int sampleSize) {
    dataRDD = trainingData;
    this.network = network;
    this.network.addRef();
    this.sampleSize = sampleSize;
    setPartitions(Math.max(1, dataRDD.sparkContext().executorEnvs().size()));
    reseed(seed);
  }
  
  /**
   * Debug.
   *
   * @param msg  the msg
   * @param args the args
   */
  protected static void debug(final @NotNull String msg, final Object... args) {
    final String format = String.format(msg, args);
    log.info(format);
  }
  
  /**
   * Gets result.
   *
   * @param delta  the delta
   * @param values the values
   * @return the result
   */
  protected static SparkTrainable.ReducableResult getResult(final @NotNull DeltaSet<NNLayer> delta, final @NotNull double[] values) {
    final Map<String, double[]> deltas = delta.getMap().entrySet().stream().collect(Collectors.toMap(
      e -> e.getKey().getId().toString(), e -> e.getValue().getDelta()
                                                                                                    ));
    return new SparkTrainable.ReducableResult(deltas, values.length, Arrays.stream(values).sum());
  }
  
  /**
   * Gets stream.
   *
   * @param partition the partition
   * @return the stream
   */
  protected static Stream<Tensor[]> getStream(final @NotNull Iterator<Tensor[]> partition) {
    final int characteristics = Spliterator.ORDERED;
    final boolean parallel = false;
    final @NotNull Spliterator<Tensor[]> spliterator = Spliterators.spliteratorUnknownSize(partition, characteristics);
    return StreamSupport.stream(spliterator, parallel);
  }
  
  /**
   * Gets delta.
   *
   * @param reduce the reduce
   * @return the delta
   */
  protected @NotNull DeltaSet<NNLayer> getDelta(final @NotNull SparkTrainable.ReducableResult reduce) {
    final @NotNull DeltaSet<NNLayer> xxx = new DeltaSet<NNLayer>();
    final Tensor[] prototype = dataRDD.toJavaRDD().take(1).get(0);
    final NNResult result = network.eval(NNConstant.batchResultArray(new Tensor[][]{prototype}));
    result.accumulate(xxx, 0);
    reduce.accumulate(xxx);
    return xxx;
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
  public @NotNull SparkTrainable setPartitions(final int partitions) {
    if (1 > partitions) throw new IllegalArgumentException();
    this.partitions = partitions;
    return this;
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
  public @NotNull SparkTrainable setStorageLevel(final StorageLevel storageLevel) {
    this.storageLevel = storageLevel;
    reseed(seed);
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
  public void setVerbose(final boolean verbose) {
    this.verbose = verbose;
  }
  
  @Override
  public PointSample measure(final TrainingMonitor monitor) {
    final long time1 = System.nanoTime();
    final JavaRDD<ReducableResult> mapPartitions = sampledRDD.toJavaRDD().mapPartitions(new PartitionTask(network));
    final long time2 = System.nanoTime();
    final SparkTrainable.ReducableResult result = mapPartitions.reduce(SparkTrainable.ReducableResult::add);
    if (isVerbose()) {
      log.info(String.format("Measure timing: %.3f / %.3f for %s items", (time2 - time1) * 1e-9, (System.nanoTime() - time2) * 1e-9, sampledRDD.count()));
    }
    final @NotNull DeltaSet<NNLayer> xxx = getDelta(result);
    return new PointSample(xxx, new StateSet<NNLayer>(xxx), result.sum, 0.0, result.count).normalize();
  }
  
  @Override
  public boolean reseed(final long seed) {
    this.seed = seed;
    assert 0 < sampleSize;
    final long count = dataRDD.count();
    assert !dataRDD.isEmpty();
    if (null != sampledRDD) {
      sampledRDD.unpersist(false);
    }
    sampledRDD = dataRDD.sample(false, sampleSize * 1.0 / count, System.currentTimeMillis())
                        .repartition(getPartitions(), null)
                        .persist(getStorageLevel());
    assert !sampledRDD.isEmpty();
    log.info(String.format("Sampled %s items from main dataset of %s (%s) items", sampledRDD.count(), count, sampleSize));
    return true;
  }
  
  @Override
  public NNLayer getLayer() {
    return network;
  }
  
  @Override
  protected void _free() {
    this.network.freeRef();
  }
  
  /**
   * The type Partition task.
   */
  @SuppressWarnings("serial")
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
    protected PartitionTask(final NNLayer network) {
      this.network = network;
    }
    
    @Override
    public @NotNull Iterator<SparkTrainable.ReducableResult> call(final @NotNull Iterator<Tensor[]> partition) throws Exception {
      final long startTime = System.nanoTime();
      final @NotNull DataTrainable trainable = new BasicTrainable(network);
      final Tensor[][] tensors = SparkTrainable.getStream(partition).toArray(i -> new Tensor[i][]);
      if (verbose) {
        SparkTrainable.debug("Materialized %s records in %4f sec", tensors.length, (System.nanoTime() - startTime) * 1e-9);
      }
      final PointSample measure = trainable.setData(Arrays.asList(tensors)).measure(new TrainingMonitor() {
        @Override
        public void log(final @NotNull String msg) {
          SparkTrainable.debug(msg);
        }
      });
      assert measure != null;
      return Arrays.asList(SparkTrainable.getResult(measure.delta, new double[]{measure.sum})).iterator();
    }
  }
  
  /**
   * The type Reducable result.
   */
  @SuppressWarnings("serial")
  protected static class ReducableResult implements Serializable {
    /**
     * The Count.
     */
    public final int count;
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
     *
     * @param deltas the deltas
     * @param count  the count
     * @param sum    the sum
     */
    public ReducableResult(final Map<String, double[]> deltas, final int count, final double sum) {
      this.deltas = deltas;
      this.count = count;
      this.sum = sum;
    }
  
    /**
     * Accumulate.
     *
     * @param source the source
     */
    public void accumulate(final @NotNull DeltaSet<NNLayer> source) {
      final Map<String, NNLayer> idIndex = source.getMap().entrySet().stream().collect(Collectors.toMap(
        e -> e.getKey().getId().toString(), e -> e.getKey()
                                                                                                       ));
      deltas.forEach((k, v) -> source.get(idIndex.get(k), (double[]) null).addInPlace(v));
    }
  
    /**
     * Add spark trainable . reducable result.
     *
     * @param right the right
     * @return the spark trainable . reducable result
     */
    public @NotNull SparkTrainable.ReducableResult add(final @NotNull SparkTrainable.ReducableResult right) {
      final @NotNull HashMap<String, double[]> map = new HashMap<>();
      final Set<String> keys = Stream.concat(deltas.keySet().stream(), right.deltas.keySet().stream()).collect(Collectors.toSet());
      for (final String key : keys) {
        final double[] l = deltas.get(key);
        final double[] r = right.deltas.get(key);
        if (null != r) {
          if (null != l) {
            assert l.length == r.length;
            final @NotNull double[] x = new double[l.length];
            for (int i = 0; i < l.length; i++) {
              x[i] = l[i] + r[i];
            }
            map.put(key, x);
          }
          else {
            map.put(key, r);
          }
        }
        else {
          assert null != l;
          map.put(key, l);
        }
      }
      return new SparkTrainable.ReducableResult(map, count + right.count, sum + right.sum);
    }
    
  }
  
}
