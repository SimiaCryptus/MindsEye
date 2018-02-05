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
import com.simiacryptus.mindseye.layers.java.PlaceholderLayer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.util.lang.TimedResult;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.stream.IntStream;

/**
 * This class handles dispatching network evaluations, and distributing the evaluations to the system GPU(s). This is
 * the main class the handles actual execution for training purposes.
 */
public class BasicTrainable extends ReferenceCountingBase implements DataTrainable, TrainableDataMask {
  
  /**
   * The Network.
   */
  protected final NNLayer network;
  /**
   * The Data.
   */
  protected List<Tensor[]> data;
  
  /**
   * The Mask.
   */
  boolean[] mask = null;
  private int verbosity = 0;
  
  /**
   * Instantiates a new Gpu trainable.
   *
   * @param network the network
   */
  public BasicTrainable(final NNLayer network) {
    this.network = network;
    this.network.addRef();
    data = null;
  }
  
  /**
   * Get nn context nn result [ ].
   *
   * @param data the data
   * @param mask the mask
   * @return the nn result [ ]
   */
  public static NNResult[] getNNContext(final List<Tensor[]> data, final boolean[] mask) {
    if (null == data) throw new IllegalArgumentException();
    if (0 >= data.size()) throw new IllegalArgumentException();
    final int cols = data.get(0).length;
    return IntStream.range(0, cols).mapToObj(col -> {
      final Tensor[] tensors = IntStream.range(0, data.size()).mapToObj(row -> data.get(row)[col]).toArray(i -> new Tensor[i]);
      TensorArray tensorArray = TensorArray.create(tensors);
      if (null == mask || col >= mask.length || !mask[col]) {
        return new NNConstant(tensorArray);
      }
      else {
        return new NNResult(tensorArray, (final DeltaSet<NNLayer> buffer, final TensorList delta) -> {
          for (int index = 0; index < delta.length(); index++) {
            final Tensor dt = delta.get(index);
            final double[] d = dt.getData();
            final Tensor t = tensors[index];
            final double[] p = t.getData();
            PlaceholderLayer<double[]> layer = new PlaceholderLayer<>(p);
            buffer.get(layer, p).addInPlace(d).freeRef();
            dt.freeRef();
            layer.freeRef();
          }
        }) {
  
          @Override
          public boolean isAlive() {
            return true;
          }
        };
      }
    }).toArray(x1 -> new NNResult[x1]);
  }
  
  /**
   * Eval point sample.
   *
   * @param list    the list
   * @param monitor the monitor
   * @return the point sample
   */
  protected PointSample eval(final List<Tensor[]> list, final TrainingMonitor monitor) {
    final TimedResult<PointSample> timedResult = TimedResult.time(() -> {
      final NNResult[] nnContext = BasicTrainable.getNNContext(list, mask);
      final NNResult result = network.eval(nnContext);
      for (NNResult nnResult : nnContext) {
        nnResult.getData().freeRef();
        nnResult.freeRef();
      }
      final TensorList resultData = result.getData();
      final DoubleSummaryStatistics statistics = resultData.stream()
                                                           .flatMapToDouble(x -> Arrays.stream(Arrays.stream(x.getData()).toArray()))
                                                           .summaryStatistics();
      final double sum = statistics.getSum();
      final DeltaSet<NNLayer> deltaSet = new DeltaSet<NNLayer>();
      result.accumulate(deltaSet, 1.0);
      resultData.freeRef();
      result.freeRef();
      //log.info(String.format("Evaluated to %s delta buffers, %s mag", DeltaSet<NNLayer>.getMap().size(), DeltaSet<NNLayer>.getMagnitude()));
      StateSet<NNLayer> stateSet = new StateSet<>(deltaSet);
      PointSample pointSample = new PointSample(deltaSet, stateSet, sum, 0.0, list.size());
      deltaSet.freeRef();
      stateSet.freeRef();
      return pointSample;
    });
    if (null != monitor && verbosity() > 0) {
      monitor.log(String.format("Device completed %s items in %.3f sec", list.size(), timedResult.timeNanos / 1e9));
    }
    PointSample normalize = timedResult.result.normalize();
    timedResult.result.freeRef();
    return normalize;
  }
  
  @Override
  public Tensor[][] getData() {
    return data.toArray(new Tensor[][]{});
  }
  
  @Override
  public boolean[] getMask() {
    return mask;
  }
  
  @Override
  public NNLayer getLayer() {
    return network;
  }
  
  /**
   * Measure point sample.
   *
   * @param monitor the monitor
   * @return the point sample
   */
  @Override
  public PointSample measure(final TrainingMonitor monitor) {
    assert !data.isEmpty();
    final TimedResult<PointSample> timedResult = TimedResult.time(() -> eval(data, monitor));
    //          log.info(String.format("Evaluated to %s delta arrays", DeltaSet<NNLayer>.run.size()));
    if (null != monitor && verbosity() > 1) {
      monitor.log(String.format("Evaluated %s items in %.4fs (%s/%s)", data.size(), timedResult.timeNanos / 1e9, timedResult.result.getMean(), timedResult.result.delta.getMagnitude()));
    }
    assert null != timedResult.result;
    return timedResult.result;
  }
  
  @Override
  public synchronized Trainable setData(final List<Tensor[]> sampledData) {
    assert !sampledData.isEmpty();
    sampledData.stream().flatMap(x -> Arrays.stream(x)).forEach(x -> x.addRef());
    if (null != this.data) this.data.stream().flatMap(x -> Arrays.stream(x)).forEach(x -> x.freeRef());
    data = sampledData;
    return this;
  }
  
  @Override
  public TrainableDataMask setMask(final boolean... mask) {
    this.mask = mask;
    return this;
  }
  
  /**
   * Sets verbose.
   *
   * @param verbose the verbose
   * @return the verbose
   */
  public BasicTrainable setVerbosity(final int verbose) {
    verbosity = verbose;
    return this;
  }
  
  /**
   * Is verbose boolean.
   *
   * @return the boolean
   */
  public int verbosity() {
    return verbosity;
  }
  
  @Override
  protected void _free() {
    this.network.freeRef();
    if (null != this.data) this.data.stream().flatMap(x -> Arrays.stream(x)).forEach(x -> x.freeRef());
  }
}
