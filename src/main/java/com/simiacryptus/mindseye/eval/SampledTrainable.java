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

/**
 * A trainable contract for variable sample size; smaller values result in faster but less representative evaluations.
 */
public interface SampledTrainable extends Trainable {
  @Override
  SampledCachedTrainable<? extends SampledTrainable> cached();
  
  /**
   * Gets training size.
   *
   * @return the training size
   */
  int getTrainingSize();
  
  /**
   * Sets training size.
   *
   * @param trainingSize the training size
   * @return the training size
   */
  SampledTrainable setTrainingSize(int trainingSize);
}
