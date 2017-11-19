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

import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.rdd.RDD;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The type Local spark trainable.
 */
public class LocalSparkTrainable extends SparkTrainable {
  
  /**
   * Instantiates a new Local spark trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   */
  public LocalSparkTrainable(RDD<Tensor[]> trainingData, NNLayer network) {
    super(trainingData, network);
  }
  
  /**
   * Instantiates a new Local spark trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param sampleSize   the sample size
   */
  public LocalSparkTrainable(RDD<Tensor[]> trainingData, NNLayer network, int sampleSize) {
    super(trainingData, network, sampleSize);
  }
  
  
  @Override
  public Trainable.PointSample measure(boolean isStatic, TrainingMonitor monitor) {
    long time1 = System.nanoTime();
    JavaRDD<Tensor[]> javaRDD = this.sampledRDD.toJavaRDD();
    assert !javaRDD.isEmpty();
    List<ReducableResult> mapPartitions = javaRDD.partitions().stream().map(partition -> {
      try {
        List<Tensor[]>[] array = javaRDD.collectPartitions(new int[]{partition.index()});
        assert 0 < array.length;
        if (0 == Arrays.stream(array).mapToInt((List<Tensor[]> x) -> x.size()).sum()) {
          return null;
        }
        assert 0 < Arrays.stream(array).mapToInt(x -> x.stream().mapToInt(y -> y.length).sum()).sum();
        Stream<Tensor[]> stream = Arrays.stream(array).flatMap(i -> i.stream());
        Iterator<Tensor[]> iterator = stream.iterator();
        return new PartitionTask(network).call(iterator).next();
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    }).filter(x -> null != x).collect(Collectors.toList());
    long time2 = System.nanoTime();
    SparkTrainable.ReducableResult result = mapPartitions.stream().reduce(SparkTrainable.ReducableResult::add).get();
    if (isVerbose()) {
      System.out.println(String.format("Measure timing: %.3f / %.3f for %s items", (time2 - time1) * 1e-9, (System.nanoTime() - time2) * 1e-9, sampledRDD.count()));
    }
    DeltaSet deltaSet = getDelta(result);
    DeltaSet stateSet = new DeltaSet();
    deltaSet.getMap().forEach((layer, layerDelta) -> {
      stateSet.get(layer, layerDelta.target).accumulate(layerDelta.target);
    });
    return new Trainable.PointSample(deltaSet, stateSet, result.sum);
  }
  
}
