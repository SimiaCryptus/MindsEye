///*
// * Copyright (c) 2018 by Andrew Charneski.
// *
// * The author licenses this file to you under the
// * Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance
// * with the License.  You may obtain a copy
// * of the License at
// *
// *   http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing,
// * software distributed under the License is distributed on an
// * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// * KIND, either express or implied.  See the License for the
// * specific language governing permissions and limitations
// * under the License.
// */
//
//package com.simiacryptus.mindseye.eval;
//
//import com.simiacryptus.mindseye.lang.DeltaSet;
//import com.simiacryptus.mindseye.lang.Layer;
//import com.simiacryptus.mindseye.lang.PointSample;
//import com.simiacryptus.mindseye.lang.StateSet;
//import com.simiacryptus.mindseye.lang.Tensor;
//import com.simiacryptus.mindseye.opt.TrainingMonitor;
//import org.apache.spark.api.java.JavaRDD;
//import org.apache.spark.rdd.RDD;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//import javax.annotation.Nonnull;
//import java.util.Arrays;
//import java.util.Iterator;
//import java.util.List;
//import java.util.stream.Collectors;
//import java.util.stream.Stream;
//
///**
// * A debugging class which replaces SparkTrainable apply an implementation that uses direct method calls instead of RMI.
// * This can be useful for debugging in some situations.
// */
//public class LocalSparkTrainable extends SparkTrainable {
//  private static final Logger log = LoggerFactory.getLogger(LocalSparkTrainable.class);
//
//  /**
//   * Instantiates a new Local spark trainable.
//   *
//   * @param trainingData the training data
//   * @param network      the network
//   */
//  public LocalSparkTrainable(final RDD<Tensor[]> trainingData, final Layer network) {
//    super(trainingData, network);
//  }
//
//  /**
//   * Instantiates a new Local spark trainable.
//   *
//   * @param trainingData the training data
//   * @param network      the network
//   * @param sampleSize   the sample size
//   */
//  public LocalSparkTrainable(final RDD<Tensor[]> trainingData, final Layer network, final int sampleSize) {
//    super(trainingData, network, sampleSize);
//  }
//
//
//  @Nonnull
//  @Override
//  public PointSample measure(final TrainingMonitor monitor) {
//    final long time1 = System.nanoTime();
//    final JavaRDD<Tensor[]> javaRDD = sampledRDD.toJavaRDD();
//    assert !javaRDD.isEmpty();
//    final List<ReducableResult> mapPartitions = javaRDD.partitions().stream().map(partition -> {
//      try {
//        final List<Tensor[]>[] array = javaRDD.collectPartitions(new int[]{partition.index()});
//        assert 0 < array.length;
//        if (0 == Arrays.stream(array).mapToInt((@Nonnull final List<Tensor[]> x) -> x.size()).sum()) {
//          return null;
//        }
//        assert 0 < Arrays.stream(array).mapToInt(x -> x.stream().mapToInt(y -> y.length).sum()).sum();
//        final Stream<Tensor[]> stream = Arrays.stream(array).flatMap(i -> i.stream());
//        @Nonnull final Iterator<Tensor[]> iterator = stream.iterator();
//        return new PartitionTask(network).call(iterator).next();
//      } catch (@Nonnull final RuntimeException e) {
//        throw e;
//      } catch (@Nonnull final Exception e) {
//        throw new RuntimeException(e);
//      }
//    }).filter(x -> null != x).collect(Collectors.toList());
//    final long time2 = System.nanoTime();
//    @Nonnull final SparkTrainable.ReducableResult result = mapPartitions.stream().reduce(SparkTrainable.ReducableResult::add).get();
//    if (isVerbose()) {
//      log.info(String.format("Measure timing: %.3f / %.3f for %s items", (time2 - time1) * 1e-9, (System.nanoTime() - time2) * 1e-9, sampledRDD.count()));
//    }
//    @Nonnull final DeltaSet<UUID> xxx = getDelta(result);
//    return new PointSample(xxx, new StateSet<UUID>(xxx), result.sum, 0.0, result.count).normalize();
//  }
//
//}
