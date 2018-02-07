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

import com.simiacryptus.util.ArrayUtil;
import org.jetbrains.annotations.NotNull;

/**
 * This strict region allows only raw scaling of weights; it is similar to but more constrained than
 * MeanVarianceGradient
 */
public class ProportionalityConstraint implements TrustRegion {
  @Override
  public @NotNull double[] project(final @NotNull double[] weights, final @NotNull double[] point) {
    return ArrayUtil.multiply(weights, ArrayUtil.dot(weights, point) / ArrayUtil.dot(weights, weights));
  }
}
