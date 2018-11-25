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

package com.simiacryptus.mindseye.test;

import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.ReferenceCounting;
import com.simiacryptus.mindseye.lang.TensorList;

import java.util.UUID;

/**
 * The interface Simple result.
 */
public interface SimpleResult extends ReferenceCounting {
  /**
   * Get derivative tensor list [ ].
   *
   * @return the tensor list [ ]
   */
  TensorList[] getInputDerivative();

  /**
   * Gets key derivative.
   *
   * @return the key derivative
   */
  DeltaSet<UUID> getLayerDerivative();


  /**
   * Gets output.
   *
   * @return the output
   */
  TensorList getOutput();
}
