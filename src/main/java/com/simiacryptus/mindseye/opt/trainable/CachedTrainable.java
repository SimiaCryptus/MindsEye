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

import java.util.ArrayList;
import java.util.List;

/**
 * The type Cached trainable.
 *
 * @param <T> the type parameter
 */
public class CachedTrainable<T> implements Trainable {
  
  /**
   * The Inner.
   */
  protected final GpuTrainable inner;
  private boolean verbose = false;
  private List<PointSample> history = new ArrayList<>();
  private int historySize = 3;
  
  /**
   * Instantiates a new Cached trainable.
   *
   * @param inner the inner
   */
  public CachedTrainable(GpuTrainable inner) {
    this.inner = inner;
  }
  
  @Override
  public PointSample measure() {
    for (PointSample result : history) {
      if (!result.weights.isDifferent()) {
        if (isVerbose()) System.out.println(String.format("Returning cached value"));
        return result;
      }
    }
    PointSample result = inner.measure();
    history.add(result.copyFull());
    while (getHistorySize() < history.size()) history.remove(0);
    return result;
  }
  
  /**
   * Is verbose boolean.
   *
   * @return the boolean
   */
  public boolean isVerbose() {
    return verbose;
  }
  
  /**
   * Sets verbose.
   *
   * @param verbose the verbose
   * @return the verbose
   */
  public CachedTrainable setVerbose(boolean verbose) {
    this.verbose = verbose;
    return this;
  }
  
  /**
   * Gets history size.
   *
   * @return the history size
   */
  public int getHistorySize() {
    return historySize;
  }
  
  /**
   * Sets history size.
   *
   * @param historySize the history size
   * @return the history size
   */
  public CachedTrainable setHistorySize(int historySize) {
    this.historySize = historySize;
    return this;
  }
  
  @Override
  public void resetToFull() {
    inner.resetToFull();
  }
  
  @Override
  public boolean resetSampling() {
    return inner.resetSampling();
  }
}
