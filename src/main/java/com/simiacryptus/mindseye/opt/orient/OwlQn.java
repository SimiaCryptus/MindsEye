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
import com.simiacryptus.mindseye.eval.Trainable.PointSample;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.synapse.JavaDenseSynapseLayer;
import com.simiacryptus.mindseye.layers.synapse.ToeplitzSynapseLayer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchPoint;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;

import java.util.Collection;
import java.util.stream.Collectors;

/**
 * The type Owl qn.
 */
public class OwlQn implements OrientationStrategy {
  /**
   * The Inner.
   */
  public final OrientationStrategy inner;
  private double factor_L1 = 0.0001;
  private double zeroTol = 1e-20;
  
  /**
   * Instantiates a new Owl qn.
   */
  public OwlQn() {
    this(new LBFGS());
  }
  
  /**
   * Instantiates a new Owl qn.
   *
   * @param inner the inner
   */
  protected OwlQn(OrientationStrategy inner) {
    this.inner = inner;
  }
  
  @Override
  public LineSearchCursor orient(Trainable subject, PointSample measurement, TrainingMonitor monitor) {
    SimpleLineSearchCursor gradient = (SimpleLineSearchCursor) inner.orient(subject, measurement, monitor);
    DeltaSet searchDirection = gradient.direction.copy();
    DeltaSet orthant = new DeltaSet();
    for (NNLayer layer : getLayers(gradient.direction.map.keySet())) {
      double[] weights = gradient.direction.map.get(layer).target;
      double[] delta = gradient.direction.map.get(layer).getDelta();
      double[] searchDir = searchDirection.get(layer, weights).getDelta();
      double[] suborthant = orthant.get(layer, weights).getDelta();
      for (int i = 0; i < searchDir.length; i++) {
        int positionSign = sign(weights[i]);
        int directionSign = sign(delta[i]);
        suborthant[i] = (0 == positionSign) ? directionSign : positionSign;
        searchDir[i] += factor_L1 * (weights[i] < 0 ? -1.0 : 1.0);
        if (sign(searchDir[i]) != directionSign) searchDir[i] = delta[i];
      }
      assert (null != searchDir);
    }
    return new SimpleLineSearchCursor(subject, measurement, searchDirection) {
      @Override
      public LineSearchPoint step(double alpha, TrainingMonitor monitor) {
        origin.weights.vector().stream().forEach(d -> d.overwrite());
        DeltaSet currentDirection = direction.copy();
        direction.map.forEach((layer, buffer) -> {
          if (null == buffer.getDelta()) return;
          double[] currentDelta = currentDirection.get(layer, buffer.target).getDelta();
          for (int i = 0; i < buffer.getDelta().length; i++) {
            double prevValue = buffer.target[i];
            double newValue = prevValue + buffer.getDelta()[i] * alpha;
            if (sign(prevValue) != 0 && sign(prevValue) != sign(newValue)) {
              currentDelta[i] = 0;
              buffer.target[i] = 0;
            }
            else {
              buffer.target[i] = newValue;
            }
          }
        });
        return new LineSearchPoint(subject.measure().setRate(alpha), dot(currentDirection.vector(), subject.measure().delta.vector()));
      }
    }.setDirectionType("OWL/QN");
  }
  
  @Override
  public void reset() {
    inner.reset();
  }
  
  /**
   * Sign int.
   *
   * @param weight the weight
   * @return the int
   */
  protected int sign(double weight) {
    if (weight > zeroTol) {
      return 1;
    }
    else if (weight < -zeroTol) {
    }
    else {
      return -1;
    }
    return 0;
  }
  
  /**
   * Gets layers.
   *
   * @param layers the layers
   * @return the layers
   */
  public Collection<NNLayer> getLayers(Collection<NNLayer> layers) {
    return layers.stream()
             .filter(layer -> {
               if (layer instanceof DenseSynapseLayer) return true;
               if (layer instanceof ToeplitzSynapseLayer) return true;
               return layer instanceof JavaDenseSynapseLayer;
             })
             .collect(Collectors.toList());
  }
  
  /**
   * Gets factor l 1.
   *
   * @return the factor l 1
   */
  public double getFactor_L1() {
    return factor_L1;
  }
  
  /**
   * Sets factor l 1.
   *
   * @param factor_L1 the factor l 1
   * @return the factor l 1
   */
  public OwlQn setFactor_L1(double factor_L1) {
    this.factor_L1 = factor_L1;
    return this;
  }
  
  /**
   * Gets zero tol.
   *
   * @return the zero tol
   */
  public double getZeroTol() {
    return zeroTol;
  }
  
  /**
   * Sets zero tol.
   *
   * @param zeroTol the zero tol
   * @return the zero tol
   */
  public OwlQn setZeroTol(double zeroTol) {
    this.zeroTol = zeroTol;
    return this;
  }
}
