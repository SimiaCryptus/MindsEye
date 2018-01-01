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

import com.simiacryptus.mindseye.lang.Tensor;

import java.util.List;

/**
 * A base class for Trainable objects advertizing an API for setting and accessing the training data.
 */
public interface DataTrainable extends Trainable {
  /**
   * Get data tensor [ ] [ ].
   *
   * @return the tensor [ ] [ ]
   */
  Tensor[][] getData();
  
  /**
   * Sets data.
   *
   * @param tensors the tensors
   * @return the data
   */
  Trainable setData(List<Tensor[]> tensors);
}
