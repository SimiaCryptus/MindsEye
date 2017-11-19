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

package com.simiacryptus.mindseye.layers.java;

/**
 * The interface Cum sum.
 */
public interface CumSum {
  /**
   * Gets carry over.
   *
   * @return the carry over
   */
  double getCarryOver();
  
  /**
   * Sets carry over.
   *
   * @param carryOver the carry over
   * @return the carry over
   */
  CumSum setCarryOver(double carryOver);
  
  /**
   * Gets carryover denominator.
   *
   * @return the carryover denominator
   */
  int getCarryoverDenominator();
  
  /**
   * Sets carryover denominator.
   *
   * @param carryoverDenominator the carryover denominator
   * @return the carryover denominator
   */
  CumSum setCarryoverDenominator(int carryoverDenominator);
}
