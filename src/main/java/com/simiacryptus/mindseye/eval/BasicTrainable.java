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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
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
  @Nullable
  protected List<Tensor[]> data;
  
  /**
   * The Mask.
   */
  @Nullable
  boolean[] mask = null;
  private int verbosity = 0;
  
  /**
   * Instantiates a new Gpu trainable.
   *
   * @param network the network
   */
  public BasicTrainable(final NNLayer network) {
    this.network = network;
    this.network.addRef(this);
    data = null;
  }
  
  /**
   * Get nn context nn result [ ].
   *
   * @param data the data
   * @param mask the mask
   * @return the nn result [ ]
   */
  public static NNResult[] getNNContext(@Nullable final List<Tensor[]> data, @Nullable final boolean[] mask) {
    if (null == data) throw new IllegalArgumentException();
    if (0 >= data.size()) throw new IllegalArgumentException();
    final int cols = data.get(0).length;
    return IntStream.range(0, cols).mapToObj(col -> {
      final Tensor[] tensors = IntStream.range(0, data.size()).mapToObj(row -> data.get(row)[col]).toArray(i -> new Tensor[i]);
      @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.create(tensors);
      if (null == mask || col >= mask.length || !mask[col]) {
        return new NNConstant(tensorArray);
      }
      else {
        return new NNResult(tensorArray, (@javax.annotation.Nonnull final DeltaSet<NNLayer> buffer, @javax.annotation.Nonnull final TensorList delta) -> {
          for (int index = 0; index < delta.length(); index++) {
            final Tensor dt = delta.get(index);
            @javax.annotation.Nullable final double[] d = dt.getData();
            final Tensor t = tensors[index];
            @javax.annotation.Nullable final double[] p = t.getData();
            @javax.annotation.Nonnull PlaceholderLayer<double[]> layer = new PlaceholderLayer<>(p);
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
  @Nonnull
  protected PointSample eval(@javax.annotation.Nonnull final List<Tensor[]> list, @Nullable final TrainingMonitor monitor) {
    @javax.annotation.Nonnull final TimedResult<PointSample> timedResult = TimedResult.time(() -> {
      final NNResult[] nnContext = BasicTrainable.getNNContext(list, mask);
      final NNResult result = network.eval(nnContext);
      for (@javax.annotation.Nonnull NNResult nnResult : nnContext) {
        nnResult.getData().freeRef();
        nnResult.freeRef();
      }
      final TensorList resultData = result.getData();
      final DoubleSummaryStatistics statistics = resultData.stream()
        .flatMapToDouble(x -> {
          double[] array = Arrays.stream(x.getData()).toArray();
          x.freeRef();
          return Arrays.stream(array);
        }).summaryStatistics();
      final double sum = statistics.getSum();
      @javax.annotation.Nonnull final DeltaSet<NNLayer> deltaSet = new DeltaSet<NNLayer>();
      result.accumulate(deltaSet, 1.0);
      resultData.freeRef();
      result.freeRef();
      //log.info(String.format("Evaluated to %s delta buffers, %s mag", DeltaSet<NNLayer>.getMap().size(), DeltaSet<NNLayer>.getMagnitude()));
      @javax.annotation.Nonnull StateSet<NNLayer> stateSet = new StateSet<>(deltaSet);
      @javax.annotation.Nonnull PointSample pointSample = new PointSample(deltaSet, stateSet, sum, 0.0, list.size());
      deltaSet.freeRef();
      stateSet.freeRef();
      return pointSample;
    });
    if (null != monitor && verbosity() > 0) {
      monitor.log(String.format("Device completed %s items in %.3f sec", list.size(), timedResult.timeNanos / 1e9));
    }
    @Nonnull PointSample normalize = timedResult.result.normalize();
    timedResult.result.freeRef();
    return normalize;
  }
  
  @javax.annotation.Nonnull
  @Override
  public Tensor[][] getData() {
    return data.toArray(new Tensor[][]{});
  }
  
  @Nullable
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
  public PointSample measure(@Nullable final TrainingMonitor monitor) {
    assert !data.isEmpty();
    @javax.annotation.Nonnull final TimedResult<PointSample> timedResult = TimedResult.time(() -> eval(data, monitor));
    //          log.info(String.format("Evaluated to %s delta arrays", DeltaSet<NNLayer>.run.size()));
    if (null != monitor && verbosity() > 1) {
      monitor.log(String.format("Evaluated %s items in %.4fs (%s/%s)", data.size(), timedResult.timeNanos / 1e9, timedResult.result.getMean(), timedResult.result.delta.getMagnitude()));
    }
    assert null != timedResult.result;
    return timedResult.result;
  }
  
  @javax.annotation.Nonnull
  @Override
  public synchronized Trainable setData(@javax.annotation.Nonnull final List<Tensor[]> data) {
    assert !data.isEmpty();
    data.stream().flatMap(x -> Arrays.stream(x)).forEach(x -> x.addRef(this));
    if (null != this.data) this.data.stream().flatMap(x -> Arrays.stream(x)).forEach(x -> x.freeRef());
    this.data = data;
    return this;
  }
  
  @javax.annotation.Nonnull
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
  @javax.annotation.Nonnull
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
