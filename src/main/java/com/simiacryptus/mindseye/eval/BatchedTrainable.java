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

import com.google.common.collect.Lists;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.opt.TrainingMonitor;

import java.util.Arrays;
import java.util.List;

/**
 * Base class to manage batched execution, where a data set is executed
 * in segments in order to manage execution memory requirements.
 */
public abstract class BatchedTrainable extends TrainableWrapper<DataTrainable> implements DataTrainable {
  
  /**
   * The Batch size.
   */
  protected final int batchSize;
  
  /**
   * Instantiates a new Batched trainable.
   *
   * @param network   the network
   * @param batchSize the batch size
   */
  public BatchedTrainable(NNLayer network, int batchSize) {
    this(new GpuTrainable(network), batchSize);
  }
  
  /**
   * Instantiates a new Batched trainable.
   *
   * @param inner     the inner
   * @param batchSize the batch size
   */
  public BatchedTrainable(DataTrainable inner, int batchSize) {
    super(inner);
    this.batchSize = batchSize;
  }
  
  @Override
  public PointSample measure(boolean isStatic, TrainingMonitor monitor) {
    List<Tensor[]> tensors = Arrays.asList(getData());
    if (batchSize < tensors.size()) {
      int batches = (int) Math.ceil(tensors.size() * 1.0 / batchSize);
      int evenBatchSize = (int) Math.ceil(tensors.size() * 1.0 / batches);
      List<List<Tensor[]>> collection = Lists.partition(tensors, evenBatchSize);
      return collection.stream().map(trainingData -> {
        if (batchSize < trainingData.size()) {
          throw new RuntimeException();
        }
        getInner().setData(trainingData);
        return super.measure(isStatic, monitor);
      }).reduce((a, b) -> a.add(b)).get();
    }
    else {
      getInner().setData(tensors);
      return super.measure(isStatic, monitor);
    }
  }
  
  
  /**
   * Gets batch size.
   *
   * @return the batch size
   */
  public int getBatchSize() {
    return batchSize;
  }
  
}
