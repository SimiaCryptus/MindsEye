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

import com.google.common.collect.Lists;
import com.simiacryptus.lang.TimedResult;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.opt.TrainingMonitor;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.List;

/**
 * Base class to manage batched execution, where a data setByCoord is executed in segments in order to manage execution
 * memory requirements.
 */
public abstract class BatchedTrainable extends TrainableWrapper<DataTrainable> implements DataTrainable {

  /**
   * The Batch size.
   */
  protected final int batchSize;
  private boolean verbose = false;

  /**
   * Instantiates a new Batched trainable.
   *
   * @param inner     the heapCopy
   * @param batchSize the batch size
   */
  public BatchedTrainable(final DataTrainable inner, final int batchSize) {
    super(inner);
    this.batchSize = batchSize;
  }

  /**
   * Instantiates a new Batched trainable.
   *
   * @param network   the network
   * @param batchSize the batch size
   */
  public BatchedTrainable(final Layer network, final int batchSize) {
    this(new BasicTrainable(network), batchSize);
    getInner().freeRef();
  }

  /**
   * Gets batch size.
   *
   * @return the batch size
   */
  public int getBatchSize() {
    return batchSize;
  }


  @Override
  public PointSample measure(final TrainingMonitor monitor) {
    @Nonnull final List<Tensor[]> tensors = Arrays.asList(getData());
    TimedResult<PointSample> timedResult = TimedResult.time(() -> {
      if (batchSize < tensors.size()) {
        final int batches = (int) Math.ceil(tensors.size() * 1.0 / batchSize);
        final int evenBatchSize = (int) Math.ceil(tensors.size() * 1.0 / batches);
        @Nonnull final List<List<Tensor[]>> collection = Lists.partition(tensors, evenBatchSize);
        return collection.stream().map(trainingData -> {
          if (batchSize < trainingData.size()) {
            throw new RuntimeException();
          }
          getInner().setData(trainingData);
          return super.measure(monitor);
        }).reduce((a, b) -> a.add(b)).get();
      } else {
        getInner().setData(tensors);
        return super.measure(monitor);
      }
    });
    if (null != monitor && isVerbose()) {
      monitor.log(String.format("Evaluated %s items in %.4fs (%s/%s)", tensors.size(), timedResult.timeNanos / 1e9, timedResult.result.getMean(), timedResult.result.delta.getMagnitude()));
    }
    return timedResult.result;
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
   * @return the verbose
   */
  public BatchedTrainable setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }
}
