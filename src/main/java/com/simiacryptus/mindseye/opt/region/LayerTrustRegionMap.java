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

package com.simiacryptus.mindseye.opt.region;

import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Andrew Charneski on 5/23/2017.
 */
public class LayerTrustRegionMap extends TrustRegionStrategy {
  private final Map<NNLayer, TrustRegion> regionPolicies = new HashMap<>();
  private TrustRegion defaultRegionPolicy = null;
  
  public TrustRegion getRegionPolicy(NNLayer layer) {
    return regionPolicies.getOrDefault(layer, defaultRegionPolicy);
  }
  
  public Map<NNLayer, TrustRegion> getRegionPolicies() {
    return regionPolicies;
  }
  
  public TrustRegion getDefaultRegionPolicy() {
    return defaultRegionPolicy;
  }
  
  public TrustRegionStrategy setDefaultRegionPolicy(TrustRegion defaultRegionPolicy) {
    this.defaultRegionPolicy = defaultRegionPolicy;
    return this;
  }
  
}
