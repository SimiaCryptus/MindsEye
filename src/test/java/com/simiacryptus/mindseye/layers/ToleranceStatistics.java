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

package com.simiacryptus.mindseye.layers;

import com.simiacryptus.util.data.DoubleStatistics;

public class ToleranceStatistics {
  public final DoubleStatistics absoluteTol;
  public final DoubleStatistics relativeTol;

  public ToleranceStatistics() {
    this(new DoubleStatistics(),new DoubleStatistics());
  }

  public ToleranceStatistics(DoubleStatistics absoluteTol, DoubleStatistics relativeTol) {
    this.absoluteTol = absoluteTol;
    this.relativeTol = relativeTol;
  }

  public ToleranceStatistics accumulate(double target, double val) {
    absoluteTol.accept(Math.abs(target-val));
    if(Double.isFinite(val+target) && val!=-target) relativeTol.accept(2 * Math.abs(target-val) / Math.abs(val+target));
    return this;
  }

  public ToleranceStatistics combine(ToleranceStatistics right) {
    return new ToleranceStatistics(
      absoluteTol.combine(right.absoluteTol),
      relativeTol.combine(right.relativeTol)
    );
  }
}
