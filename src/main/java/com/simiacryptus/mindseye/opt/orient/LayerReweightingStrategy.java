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

package com.simiacryptus.mindseye.opt.orient;

import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.DoubleBuffer;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;

import java.util.HashMap;

import static com.simiacryptus.util.ArrayUtil.multiply;

/**
 * This wrapping strategy alters the (Simple)LineCursor returned by the inner strategy
 * to effectively tune the learning rate for each layer.
 */
public abstract class LayerReweightingStrategy implements OrientationStrategy<SimpleLineSearchCursor> {
  
  /**
   * The Inner.
   */
  public final OrientationStrategy<SimpleLineSearchCursor> inner;
  
  
  /**
   * Instantiates a new Layer reweighting strategy.
   *
   * @param inner the inner
   */
  public LayerReweightingStrategy(OrientationStrategy<SimpleLineSearchCursor> inner) {
    this.inner = inner;
  }
  
  @Override
  public SimpleLineSearchCursor orient(Trainable subject, PointSample measurement, TrainingMonitor monitor) {
    SimpleLineSearchCursor orient = inner.orient(subject, measurement, monitor);
    DeltaSet direction = ((SimpleLineSearchCursor) orient).direction;
    direction.getMap().forEach((layer, buffer) -> {
      if (null == buffer.getDelta()) return;
      Double weight = getRegionPolicy(layer);
      if (null != weight && 0 < weight) {
        DoubleBuffer deltaBuffer = direction.get(layer, buffer.target);
        double[] adjusted = multiply(deltaBuffer.getDelta(), weight);
        for (int i = 0; i < adjusted.length; i++) {
          deltaBuffer.getDelta()[i] = adjusted[i];
        }
      }
    });
    return orient;
  }
  
  /**
   * Gets region policy.
   *
   * @param layer the layer
   * @return the region policy
   */
  public abstract Double getRegionPolicy(NNLayer layer);
  
  /**
   * The type Hash map layer reweighting strategy.
   */
  public static class HashMapLayerReweightingStrategy extends LayerReweightingStrategy {
    
    private final HashMap<NNLayer, Double> map = new HashMap<>();
  
    /**
     * Instantiates a new Hash map layer reweighting strategy.
     *
     * @param inner the inner
     */
    public HashMapLayerReweightingStrategy(OrientationStrategy inner) {
      super(inner);
    }
    
    @Override
    public Double getRegionPolicy(NNLayer layer) {
      return getMap().get(layer);
    }
  
    /**
     * Gets map.
     *
     * @return the map
     */
    public HashMap<NNLayer, Double> getMap() {
      return map;
    }
    
    @Override
    public void reset() {
      inner.reset();
    }
  }
  
}
