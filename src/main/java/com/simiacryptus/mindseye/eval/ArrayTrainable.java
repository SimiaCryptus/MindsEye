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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import org.jetbrains.annotations.NotNull;

import java.util.List;

/**
 * Basic training component which evaluates a static array of data on a network. Evaluation is subject to batch-size
 * conditions to manage execution memory requirements.
 */
public class ArrayTrainable extends BatchedTrainable implements TrainableDataMask {
  
  private Tensor[][] trainingData;
  
  /**
   * Instantiates a new Array trainable.
   *
   * @param inner        the heapCopy
   * @param trainingData the training data
   */
  public ArrayTrainable(DataTrainable inner, @NotNull Tensor[]... trainingData) {
    this(inner, trainingData, trainingData.length);
  }
  
  /**
   * Instantiates a new Array trainable.
   *
   * @param inner        the heapCopy
   * @param trainingData the training data
   * @param batchSize    the batch size
   */
  public ArrayTrainable(DataTrainable inner, @NotNull Tensor[][] trainingData, int batchSize) {
    super(inner, batchSize);
    this.trainingData = trainingData;
    for (@NotNull Tensor[] tensors : trainingData) {
      for (@NotNull Tensor tensor : tensors) {
        tensor.addRef();
      }
    }
  }
  
  /**
   * Instantiates a new Array trainable.
   *
   * @param network   the network
   * @param batchSize the batch size
   */
  public ArrayTrainable(final NNLayer network, final int batchSize) {
    this(null, network, batchSize);
  }
  
  /**
   * Instantiates a new Array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   */
  public ArrayTrainable(final @NotNull Tensor[][] trainingData, final NNLayer network) {
    this(trainingData, network, trainingData.length);
  }
  
  /**
   * Instantiates a new Array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param batchSize    the batch size
   */
  public ArrayTrainable(final @NotNull Tensor[][] trainingData, final NNLayer network, final int batchSize) {
    super(network, batchSize);
    this.trainingData = trainingData;
    for (@NotNull Tensor[] tensors : trainingData) {
      for (@NotNull Tensor tensor : tensors) {
        tensor.addRef();
      }
    }
  }
  
  @Override
  public Tensor[][] getData() {
    return trainingData;
  }
  
  @Override
  protected void _free() {
    for (@NotNull Tensor[] tensors : trainingData) {
      for (@NotNull Tensor tensor : tensors) {
        tensor.freeRef();
      }
    }
    super._free();
  }
  
  @Override
  public @NotNull Trainable setData(final @NotNull List<Tensor[]> tensors) {
    for (@NotNull Tensor[] ts : tensors) {
      for (@NotNull Tensor tensor : ts) {
        tensor.addRef();
      }
    }
    if (null != trainingData) for (@NotNull Tensor[] ts : trainingData) {
      for (@NotNull Tensor tensor : ts) {
        tensor.freeRef();
      }
    }
    trainingData = tensors.toArray(new Tensor[][]{});
    return this;
  }
  
  /**
   * Sets training data.
   *
   * @param tensors the training data
   */
  public void setTrainingData(final @NotNull Tensor[][] tensors) {
    for (@NotNull Tensor[] ts : tensors) {
      for (@NotNull Tensor tensor : ts) {
        tensor.addRef();
      }
    }
    if (null != trainingData) for (@NotNull Tensor[] ts : trainingData) {
      for (@NotNull Tensor tensor : ts) {
        tensor.freeRef();
      }
    }
    this.trainingData = tensors;
  }
  
  @Override
  public @NotNull ArrayTrainable setMask(boolean... mask) {
    return (ArrayTrainable) super.setMask(mask);
  }
}
