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

package com.simiacryptus.mindseye.layers.activation;

@SuppressWarnings("serial")
public class MaxConstLayer extends SimpleActivationLayer<MaxConstLayer> {
  
  private double value = 0;
  
  @Override
  protected void eval(final double x, final double[] results) {
    final double d = x < this.value ? 0 : 1;
    final double f = x < this.value ? this.value : x;
    assert Double.isFinite(d);
    results[0] = f;
    results[1] = d;
  }
  
  public double getValue() {
    return this.value;
  }
  
  public MaxConstLayer setValue(final double value) {
    this.value = value;
    return this;
  }
}
