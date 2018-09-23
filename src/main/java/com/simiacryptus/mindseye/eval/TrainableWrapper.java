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

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.lang.ReferenceCountingBase;
import com.simiacryptus.mindseye.opt.TrainingMonitor;

import javax.annotation.Nonnull;

/**
 * A base class for a Trainable type which wraps an heapCopy type of the same kind.
 *
 * @param <T> the type parameter
 */
public class TrainableWrapper<T extends Trainable> extends ReferenceCountingBase implements TrainableDataMask {

  private final T inner;

  /**
   * Instantiates a new Trainable wrapper.
   *
   * @param inner the heapCopy
   */
  public TrainableWrapper(final T inner) {
    this.inner = inner;
    this.inner.addRef(this);
  }

  @Override
  protected void _free() {
    this.inner.freeRef();
    super._free();
  }

  /**
   * Gets heapCopy.
   *
   * @return the heapCopy
   */
  public T getInner() {
    return inner;
  }

  @Override
  public boolean[] getMask() {
    return ((TrainableDataMask) inner).getMask();
  }

  @Override
  public PointSample measure(final TrainingMonitor monitor) {
    return inner.measure(monitor);
  }

  @Override
  public boolean reseed(final long seed) {
    return getInner().reseed(seed);
  }

  @Override
  public Layer getLayer() {
    return inner.getLayer();
  }

  @Nonnull
  @Override
  public TrainableDataMask setMask(final boolean... mask) {
    ((TrainableDataMask) inner).setMask(mask);
    return this;
  }

  @Nonnull
  @Override
  public String toString() {
    return getClass().getSimpleName() + "{" +
        "heapCopy=" + inner +
        '}';
  }
}
