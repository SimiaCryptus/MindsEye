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

package com.simiacryptus.mindseye.lang;

/**
 * The type Delta.
 */
@SuppressWarnings({"rawtypes", "unchecked"})
public class State extends DoubleBuffer {
  
  
  /**
   * Instantiates a new Delta.
   *
   * @param target the target
   * @param delta  the delta
   * @param layer  the layer
   */
  public State(final double[] target, final double[] delta, final NNLayer layer) {
    super(target,delta,layer);
  }
  
  public State(final double[] target, final NNLayer layer) {
    super(target,layer);
  }
  
  public final synchronized DoubleBuffer backup() {
    double[] delta = getDelta();
    System.arraycopy(target,0,delta,0,target.length);
    return this;
  }
  
  /**
   * Overwrite.
   */
  public final synchronized DoubleBuffer restore() {
    System.arraycopy(delta,0,target,0,target.length);
    return this;
  }
  
  public State accumulate(double[] delta) {
    Delta.accumulate(this.delta, delta, null);
    return this;
  }
}
