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

import com.simiacryptus.lang.NotImplementedException;

/**
 * The base class for the component representing the trust region volumes used for optimization. This interface provides
 * optional use of a position history, describing previous iterations' position, along apply the current position. These
 * states can be used to define the relevant trust region. The trust region is implemented as a projection function
 * which ensures a candidate state is either within the trust region volume, or on the boundary. If projection is
 * needed, it must end up at the boundary and it must be the closest point, apply the vector from the input to point
 * point being normal to the trust region surface at the position of the output.
 */
public interface TrustRegion {
  /**
   * Project double [ ].
   *
   * @param state the state
   * @param point the point
   * @return the double [ ]
   */
  default double[] project(final double[] state, final double[] point) {
    throw new NotImplementedException();
  }

  /**
   * Project double [ ].
   *
   * @param history the history
   * @param point   the point
   * @return the double [ ]
   */
  default double[] project(final double[][] history, final double[] point) {
    return project(history[0], point);
  }
}
