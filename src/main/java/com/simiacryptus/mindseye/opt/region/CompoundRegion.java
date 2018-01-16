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

package com.simiacryptus.mindseye.opt.region;

/**
 * This combination region yields the effective itersection of the trust region volumes.
 */
public class CompoundRegion implements TrustRegion {
  
  private final TrustRegion[] inner;
  
  /**
   * Instantiates a new Compound region.
   *
   * @param inner the localCopy
   */
  public CompoundRegion(final TrustRegion... inner) {
    this.inner = inner;
  }
  
  @Override
  public double[] project(final double[][] history, final double[] point) {
    double[] returnValue = point;
    for (int i = 0; i < inner.length; i++) {
      returnValue = inner[i].project(history, returnValue);
    }
    return returnValue;
  }
  
}
