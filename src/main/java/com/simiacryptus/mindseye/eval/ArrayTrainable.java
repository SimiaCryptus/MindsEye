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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.Tensor;

import java.util.List;

/**
 * The type Array trainable.
 */
public class ArrayTrainable extends BatchedTrainable implements TrainableDataMask {
  
  private Tensor[][] trainingData;
  
  /**
   * Instantiates a new Array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   */
  public ArrayTrainable(Tensor[][] trainingData, NNLayer network) {
    this(trainingData, network, trainingData.length);
  }
  
  /**
   * Instantiates a new Static array trainable.
   *
   * @param network   the network
   * @param batchSize the batch size
   */
  public ArrayTrainable(NNLayer network, int batchSize) {
    this(null, network, batchSize);
  }
  
  /**
   * Instantiates a new Array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param batchSize    the batch size
   */
  public ArrayTrainable(Tensor[][] trainingData, NNLayer network, int batchSize) {
    super(network, batchSize);
    this.trainingData = trainingData;
  }
  
  @Override
  public Trainable setData(List<Tensor[]> tensors) {
    trainingData = tensors.toArray(new Tensor[][]{});
    return this;
  }
  
  @Override
  public Tensor[][] getData() {
    return trainingData;
  }
  
  /**
   * Sets training data.
   *
   * @param trainingData the training data
   */
  public void setTrainingData(Tensor[][] trainingData) {
    this.trainingData = trainingData;
  }
}
