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

package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.synapse.JavaDenseSynapseLayer;
import com.simiacryptus.mindseye.net.synapse.ToeplitzSynapseLayer;

import java.util.Collection;
import java.util.stream.Collectors;

public class PerLayerTrustRegions implements OrientationStrategy {
  public final OrientationStrategy inner;
  private double factor_L1 = 0.0001;
  private double zeroTol = 1e-20;
  
  public PerLayerTrustRegions() {
    this(new LBFGS());
  }
  
  protected PerLayerTrustRegions(OrientationStrategy inner) {
    this.inner = inner;
  }
  
  @Override
  public LineSearchCursor orient(Trainable subject, Trainable.PointSample measurement, TrainingMonitor monitor) {
    SimpleLineSearchCursor gradient = (SimpleLineSearchCursor) inner.orient(subject, measurement, monitor);
    DeltaSet searchDirection = gradient.direction.copy();
    DeltaSet orthant = new DeltaSet();
    for (NNLayer layer : getLayers(gradient.direction.map.keySet())) {
      double[] weights = gradient.direction.map.get(layer).target;
      double[] delta = gradient.direction.map.get(layer).delta;
      double[] searchDir = searchDirection.get(layer, weights).delta;
      double[] suborthant = orthant.get(layer, weights).delta;
      for (int i = 0; i < searchDir.length; i++) {
        int positionSign = sign(weights[i]);
        int directionSign = sign(delta[i]);
        suborthant[i] = (0 == positionSign) ? directionSign : positionSign;
        searchDir[i] += factor_L1 * (weights[i] < 0 ? -1.0 : 1.0);
        if (sign(searchDir[i]) != directionSign) searchDir[i] = delta[i];
      }
      assert (null != searchDir);
    }
    return new SimpleLineSearchCursor(measurement, searchDirection, subject) {
      @Override
      public LineSearchPoint step(double alpha, TrainingMonitor monitor) {
        origin.weights.vector().stream().forEach(d -> d.overwrite());
        DeltaSet currentDirection = direction.copy();
        direction.map.forEach((layer, buffer) -> {
          if (null == buffer.delta) return;
          double[] currentDelta = currentDirection.get(layer, buffer.target).delta;
          for (int i = 0; i < buffer.delta.length; i++) {
            double prevValue = buffer.target[i];
            double newValue = prevValue + buffer.delta[i] * alpha;
            if (sign(prevValue) != 0 && sign(prevValue) != sign(newValue)) {
              currentDelta[i] = 0;
              buffer.target[i] = 0;
            } else {
              buffer.target[i] = newValue;
            }
          }
        });
        return new LineSearchPoint(subject.measure(), dot(currentDirection.vector(), subject.measure().delta.vector()));
      }
    };
  }
  
  protected int sign(double weight) {
    if (weight > zeroTol) {
      return 1;
    } else if (weight < -zeroTol) {
    } else {
      return -1;
    }
    return 0;
  }
  
  public Collection<NNLayer> getLayers(Collection<NNLayer> layers) {
    return layers.stream()
               .filter(layer -> {
                 if (layer instanceof DenseSynapseLayer) return true;
                 if (layer instanceof ToeplitzSynapseLayer) return true;
                 return layer instanceof JavaDenseSynapseLayer;
               })
               .collect(Collectors.toList());
  }
  
  public double getFactor_L1() {
    return factor_L1;
  }
  
  public PerLayerTrustRegions setFactor_L1(double factor_L1) {
    this.factor_L1 = factor_L1;
    return this;
  }
  
  public double getZeroTol() {
    return zeroTol;
  }
  
  public PerLayerTrustRegions setZeroTol(double zeroTol) {
    this.zeroTol = zeroTol;
    return this;
  }
}
