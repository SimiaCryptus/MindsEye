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

import com.simiacryptus.mindseye.data.Tensor;
import com.simiacryptus.mindseye.data.TensorList;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
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
 * The type Spark trainable. Calculation happens locally without Spark serialization overhead or multi-node compute capability
 */
public class LocalSparkTrainable extends SparkTrainable {
  
  public LocalSparkTrainable(RDD<Tensor[]> trainingData, NNLayer network) {
    super(trainingData, network);
  }
  
  public LocalSparkTrainable(RDD<Tensor[]> trainingData, NNLayer network, int sampleSize) {
    super(trainingData, network, sampleSize);
  }
  
  
  @Override
  public Trainable.PointSample measure() {
    long time1 = System.nanoTime();
    JavaRDD<Tensor[]> javaRDD = this.sampledRDD.toJavaRDD();
    List<ReducableResult> mapPartitions = javaRDD.partitions().stream().map(partition -> {
      try {
        return new PartitionTask(network).call(Arrays.stream(javaRDD.collectPartitions(new int[]{partition.index()})).flatMap(i -> i.stream()).iterator()).next();
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    }).collect(Collectors.toList());
    long time2 = System.nanoTime();
    SparkTrainable.ReducableResult result = mapPartitions.stream().reduce(SparkTrainable.ReducableResult::add).get();
    if(isVerbose()) System.out.println(String.format("Measure timing: %.3f / %.3f", (time2 - time1) * 1e-9, (System.nanoTime() - time2) * 1e-9));
    DeltaSet deltaSet = getDelta(result);
    DeltaSet stateSet = new DeltaSet();
    deltaSet.map.forEach((layer, layerDelta) -> {
      stateSet.get(layer, layerDelta.target).accumulate(layerDelta.target);
    });
    return new Trainable.PointSample(deltaSet, stateSet, result.sum);
  }
  
}
