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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.NNLayer;

/**
 * The type Array trainable.
 */
public class StaticArrayTrainable extends ArrayTrainable {
  
  private Tensor[][] trainingData;
  
  /**
   * Instantiates a new Array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   */
  public StaticArrayTrainable(Tensor[][] trainingData, NNLayer network) {
    this(trainingData, network, trainingData.length);
  }
  
  /**
   * Instantiates a new Array trainable.
   *
   * @param trainingData the training data
   * @param network      the network
   * @param batchSize    the batch size
   */
  public StaticArrayTrainable(Tensor[][] trainingData, NNLayer network, int batchSize) {
    super(network, batchSize);
    if (0 == trainingData.length) throw new IllegalArgumentException();
    this.setTrainingData(trainingData);
    resetSampling();
  }
  
  @Override
  public Tensor[][] getTrainingData() {
    return trainingData;
  }
  
  public void setTrainingData(Tensor[][] trainingData) {
    this.trainingData = trainingData;
  }
}
