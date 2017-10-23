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

public abstract class TrainableWrapper<T extends Trainable> implements Trainable {
  
  private final T inner;
  
  public TrainableWrapper(T inner) {
    this.inner = inner;
  }
  
  @Override
  public PointSample measure() {
    return inner.measure();
  };
  
  public T getInner() {
    return inner;
  }
  
  @Override
  public void resetToFull() {
    getInner().resetToFull();
  }
  
  @Override
  public boolean resetSampling() {
    return getInner().resetSampling();
  }
  
}
