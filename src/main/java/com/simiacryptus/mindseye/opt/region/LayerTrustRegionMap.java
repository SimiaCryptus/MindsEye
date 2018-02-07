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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy;
import org.jetbrains.annotations.Nullable;

import java.util.HashMap;
import java.util.Map;

/**
 * A concrete class of the TrustRegion orientation strategy base class, this uses a map collection to store per-layer
 * Trust Region configurations.
 */
public class LayerTrustRegionMap extends TrustRegionStrategy {
  private final Map<NNLayer, TrustRegion> regionPolicies = new HashMap<>();
  private @Nullable TrustRegion defaultRegionPolicy = null;
  
  /**
   * Gets default region policy.
   *
   * @return the default region policy
   */
  public @Nullable TrustRegion getDefaultRegionPolicy() {
    return defaultRegionPolicy;
  }
  
  /**
   * Gets region policies.
   *
   * @return the region policies
   */
  @javax.annotation.Nonnull
  public Map<NNLayer, TrustRegion> getRegionPolicies() {
    return regionPolicies;
  }
  
  @Override
  public TrustRegion getRegionPolicy(final NNLayer layer) {
    return regionPolicies.getOrDefault(layer, defaultRegionPolicy);
  }
  
  @Override
  public void reset() {
    inner.reset();
  }
  
  /**
   * Sets default region policy.
   *
   * @param defaultRegionPolicy the default region policy
   * @return the default region policy
   */
  @javax.annotation.Nonnull
  public TrustRegionStrategy setDefaultRegionPolicy(final TrustRegion defaultRegionPolicy) {
    this.defaultRegionPolicy = defaultRegionPolicy;
    return this;
  }
}
