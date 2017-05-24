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

import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.ml.Tensor;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.rdd.RDD;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public class SparkTrainable implements Trainable {
  
  private static class ReducableResult implements Serializable {
    public final Map<String, double[]> deltas;
    public final double sum;
    public final int count;
    
    public ReducableResult(Map<String, double[]> deltas, double sum, int count) {
      this.deltas = deltas;
      this.sum = sum;
      this.count = count;
    }
    
    public void accumulate(DeltaSet source) {
      Map<String, NNLayer> idIndex = source.map.entrySet().stream().collect(Collectors.toMap(
          e -> e.getKey().id.toString(), e -> e.getKey()
      ));
      deltas.forEach((k,v)->source.get(idIndex.get(k), (double[])null).accumulate(v));
    }
    
    public ReducableResult add(ReducableResult right) {
      HashMap<String, double[]> map = new HashMap<>();
      Set<String> keys = Stream.concat(deltas.keySet().stream(), right.deltas.keySet().stream()).collect(Collectors.toSet());
      for(String key : keys) {
        double[] l = deltas.get(key);
        double[] r = right.deltas.get(key);
        if(null != r) {
          if(null != l) {
            assert(l.length==r.length);
            double[] x = new double[l.length];
            for(int i=0;i<l.length;i++) x[i] = l[i]+r[i];
            map.put(key, x);
          } else {
            map.put(key, r);
          }
        } else {
          assert(null != l);
          map.put(key, l);
        }
      }
      return new ReducableResult(map, sum+right.sum, count+right.count);
    }
    
    public double meanValue() {
      return sum / count;
    }
  }
  
  private static class PartitionTask implements FlatMapFunction<Iterator<Tensor[]>, ReducableResult> {
    final DAGNetwork network;
    
    private PartitionTask(DAGNetwork network) {
      this.network = network;
    }
    
    @Override
    public Iterator<ReducableResult> call(Iterator<Tensor[]> partition) throws Exception {
      Tensor[][] tensors = SparkTrainable.getStream(partition).toArray(i -> new Tensor[i][]);
      NNResult eval = network.eval(NNResult.batchResultArray(tensors));
      DeltaSet deltaSet = new DeltaSet();
      eval.accumulate(deltaSet);
      double[] doubles = Arrays.stream(eval.data).mapToDouble(x -> x.getData()[0]).toArray();
      return Arrays.asList(SparkTrainable.getResult(deltaSet, doubles)).iterator();
    }
  }
  private static ReducableResult getResult(DeltaSet delta, double[] values) {
    Map<String, double[]> deltas = delta.map.entrySet().stream().collect(Collectors.toMap(
        e -> e.getKey().id.toString(), e -> e.getValue().delta
    ));
    return new ReducableResult(deltas, Arrays.stream(values).sum(), values.length);
  }
  
  private DeltaSet getDelta(ReducableResult reduce) {
    DeltaSet deltaSet = new DeltaSet();
    Tensor[] prototype = dataRDD.take(1).get(0);
    NNResult result = network.eval(NNResult.batchResultArray(new Tensor[][]{prototype}));
    result.accumulate(deltaSet, 0);
    reduce.accumulate(deltaSet);
    return deltaSet;
  }
  
  private final JavaRDD<Tensor[]> dataRDD;
  private final DAGNetwork network;
  
  public SparkTrainable(RDD<Tensor[]> trainingData, DAGNetwork network) {
    this.dataRDD = trainingData.toJavaRDD();
    this.network = network;
    resetSampling();
  }
  
  @Override
  public PointSample measure() {
    ReducableResult result = dataRDD.mapPartitions(new PartitionTask(network))
                                 .reduce(ReducableResult::add);
    DeltaSet deltaSet = getDelta(result);
    DeltaSet stateSet = new DeltaSet();
    deltaSet.map.forEach((layer, layerDelta) -> {
      stateSet.get(layer, layerDelta.target).accumulate(layerDelta.target);
    });
    return new PointSample(deltaSet, stateSet, result.meanValue());
  }
  
  @Override
  public void resetToFull() {
  }
  
  private static Stream<Tensor[]> getStream(Iterator<Tensor[]> partition) {
    int characteristics = Spliterator.ORDERED;
    boolean parallel = false;
    Spliterator<Tensor[]> spliterator = Spliterators.spliteratorUnknownSize(partition, characteristics);
    return StreamSupport.stream(spliterator, parallel);
  }
  
  @Override
  public void resetSampling() {
  }
  
}
