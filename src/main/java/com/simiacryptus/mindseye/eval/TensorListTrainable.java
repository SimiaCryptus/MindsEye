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

import com.simiacryptus.mindseye.lang.ConstantResult;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.lang.ReferenceCountingBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.StateSet;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.layers.java.PlaceholderLayer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.util.lang.TimedResult;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.stream.IntStream;

/**
 * This class handles dispatching network evaluations, and distributing the evaluations to the system GPU(s). This is
 * the main class the handles actual execution for training purposes.
 */
public class TensorListTrainable extends ReferenceCountingBase implements TrainableDataMask {
  
  /**
   * The Network.
   */
  protected final Layer network;
  /**
   * The Data.
   */
  @Nullable
  protected TensorList[] data;
  
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
   * @param data    the data
   */
  public TensorListTrainable(final Layer network, final TensorList... data) {
    this.network = network;
    this.network.addRef(this);
    this.data = data;
  }
  
  /**
   * Get nn context nn result [ ].
   *
   * @param data the data
   * @param mask the mask
   * @return the nn result [ ]
   */
  public static Result[] getNNContext(@Nullable final TensorList[] data, @Nullable final boolean[] mask) {
    if (null == data) throw new IllegalArgumentException();
    int inputs = data.length;
    assert 0 < inputs;
    int items = data[0].length();
    assert 0 < items;
    return IntStream.range(0, inputs).mapToObj(col -> {
      final Tensor[] tensors = IntStream.range(0, items).mapToObj(row -> data[col].get(row)).toArray(i -> new Tensor[i]);
      @Nonnull TensorArray tensorArray = TensorArray.create(tensors);
      if (null == mask || col >= mask.length || !mask[col]) {
        return new ConstantResult(tensorArray);
      }
      else {
        return new Result(tensorArray, (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
          for (int index = 0; index < delta.length(); index++) {
            final Tensor dt = delta.get(index);
            @Nullable final double[] d = dt.getData();
            final Tensor t = tensors[index];
            @Nullable final double[] p = t.getData();
            @Nonnull PlaceholderLayer<double[]> layer = new PlaceholderLayer<>(p);
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
    }).toArray(x1 -> new Result[x1]);
  }
  
  /**
   * Eval point sample.
   *
   * @param list    the list
   * @param monitor the monitor
   * @return the point sample
   */
  @Nonnull
  protected PointSample eval(@Nonnull final TensorList[] list, @Nullable final TrainingMonitor monitor) {
    int inputs = data.length;
    assert 0 < inputs;
    int items = data[0].length();
    assert 0 < items;
    @Nonnull final TimedResult<PointSample> timedResult = TimedResult.time(() -> {
      final Result[] nnContext = TensorListTrainable.getNNContext(list, mask);
      final Result result = network.eval(nnContext);
      for (@Nonnull Result nnResult : nnContext) {
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
      @Nonnull final DeltaSet<Layer> deltaSet = new DeltaSet<Layer>();
      @Nonnull PointSample pointSample;
      try {
        result.accumulate(deltaSet, 1.0);
        //log.info(String.format("Evaluated to %s evalInputDelta buffers, %s mag", DeltaSet<LayerBase>.getMap().size(), DeltaSet<LayerBase>.getMagnitude()));
        @Nonnull StateSet<Layer> stateSet = new StateSet<>(deltaSet);
        pointSample = new PointSample(deltaSet, stateSet, sum, 0.0, items);
        stateSet.freeRef();
      } finally {
        resultData.freeRef();
        result.freeRef();
        deltaSet.freeRef();
      }
      return pointSample;
    });
    if (null != monitor && verbosity() > 0) {
      monitor.log(String.format("Device completed %s items in %.3f sec", items, timedResult.timeNanos / 1e9));
    }
    @Nonnull PointSample normalize = timedResult.result.normalize();
    timedResult.result.freeRef();
    return normalize;
  }
  
  /**
   * Get data tensor list [ ].
   *
   * @return the tensor list [ ]
   */
  @Nonnull
  public TensorList[] getData() {
    return data;
  }
  
  @Nullable
  @Override
  public boolean[] getMask() {
    return mask;
  }
  
  @Override
  public Layer getLayer() {
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
    int inputs = data.length;
    assert 0 < inputs;
    int items = data[0].length();
    assert 0 < items;
    @Nonnull final TimedResult<PointSample> timedResult = TimedResult.time(() -> eval(data, monitor));
    //          log.info(String.format("Evaluated to %s evalInputDelta arrays", DeltaSet<LayerBase>.apply.size()));
    if (null != monitor && verbosity() > 1) {
      monitor.log(String.format("Evaluated %s items in %.4fs (%s/%s)", items, timedResult.timeNanos / 1e9, timedResult.result.getMean(), timedResult.result.delta.getMagnitude()));
    }
    assert null != timedResult.result;
    return timedResult.result;
  }
  
  /**
   * Sets data.
   *
   * @param data the data
   * @return the data
   */
  @Nonnull
  public synchronized Trainable setData(@Nonnull final TensorList[] data) {
    int inputs = data.length;
    assert 0 < inputs;
    int items = data[0].length();
    assert 0 < items;
    Arrays.stream(data).forEach(x -> x.addRef(this));
    this.data = data;
    return this;
  }
  
  @Nonnull
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
  @Nonnull
  public TensorListTrainable setVerbosity(final int verbose) {
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
    if (null != this.data) Arrays.stream(this.data).forEach(x -> x.freeRef());
  }
}
