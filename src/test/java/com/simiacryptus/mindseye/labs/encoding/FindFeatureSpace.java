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

package com.simiacryptus.mindseye.labs.encoding;

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.util.io.NotebookOutput;

import java.util.stream.Stream;

/**
 * The type Find feature space.
 */
public abstract class FindFeatureSpace {
  /**
   * The Log.
   */
  protected final NotebookOutput log;
  /**
   * The Input bands.
   */
  protected final int inputBands;
  /**
   * The Averages.
   */
  protected double[] averages;
  /**
   * The Vectors.
   */
  protected Tensor[] vectors;
  
  /**
   * Instantiates a new Find feature space.
   *
   * @param log        the log
   * @param inputBands the input bands
   */
  public FindFeatureSpace(NotebookOutput log, int inputBands) {
    this.log = log;
    this.inputBands = inputBands;
  }
  
  /**
   * Get averages double [ ].
   *
   * @return the double [ ]
   */
  public double[] getAverages() {
    return averages;
  }
  
  /**
   * Get vectors tensor [ ].
   *
   * @return the tensor [ ]
   */
  public Tensor[] getVectors() {
    return vectors;
  }
  
  /**
   * Invoke find feature space.
   *
   * @return the find feature space
   */
  public abstract FindFeatureSpace invoke();
  
  /**
   * The Features.
   *
   * @return the features
   */
  public abstract Stream<Tensor[]> getFeatures();
}
