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

package com.simiacryptus.mindseye.opt.orient;

import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchPoint;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;

import java.util.Collection;
import java.util.stream.Collectors;

/**
 * Orthant-Wise Limited-memory Quasi-Newton optimization This is a modified L-BFGS algorithm which uses orthant trust
 * regions to bound the cursor path during the line search phase of each iteration
 */
public class OwlQn implements OrientationStrategy<LineSearchCursor> {
  /**
   * The Inner.
   */
  public final OrientationStrategy<?> inner;
  private double factor_L1 = 0.000;
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
  protected OwlQn(final OrientationStrategy<?> inner) {
    this.inner = inner;
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
  public OwlQn setFactor_L1(final double factor_L1) {
    this.factor_L1 = factor_L1;
    return this;
  }
  
  /**
   * Gets layers.
   *
   * @param layers the layers
   * @return the layers
   */
  public Collection<NNLayer> getLayers(final Collection<NNLayer> layers) {
    return layers.stream()
                 .filter(layer -> {
                   return layer instanceof FullyConnectedLayer;
                 })
                 .collect(Collectors.toList());
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
  public OwlQn setZeroTol(final double zeroTol) {
    this.zeroTol = zeroTol;
    return this;
  }
  
  @Override
  public LineSearchCursor orient(final Trainable subject, final PointSample measurement, final TrainingMonitor monitor) {
    final SimpleLineSearchCursor gradient = (SimpleLineSearchCursor) inner.orient(subject, measurement, monitor);
    final DeltaSet<NNLayer> searchDirection = gradient.direction.copy();
    final DeltaSet<NNLayer> orthant = new DeltaSet<NNLayer>();
    for (final NNLayer layer : getLayers(gradient.direction.getMap().keySet())) {
      final double[] weights = gradient.direction.getMap().get(layer).target;
      final double[] delta = gradient.direction.getMap().get(layer).getDelta();
      final double[] searchDir = searchDirection.get(layer, weights).getDelta();
      final double[] suborthant = orthant.get(layer, weights).getDelta();
      for (int i = 0; i < searchDir.length; i++) {
        final int positionSign = sign(weights[i]);
        final int directionSign = sign(delta[i]);
        suborthant[i] = 0 == positionSign ? directionSign : positionSign;
        searchDir[i] += factor_L1 * (weights[i] < 0 ? -1.0 : 1.0);
        if (sign(searchDir[i]) != directionSign) {
          searchDir[i] = delta[i];
        }
      }
      assert null != searchDir;
    }
    return new SimpleLineSearchCursor(subject, measurement, searchDirection) {
      @Override
      public LineSearchPoint step(final double alpha, final TrainingMonitor monitor) {
        origin.weights.stream().forEach(d -> d.restore());
        final DeltaSet<NNLayer> currentDirection = direction.copy();
        direction.getMap().forEach((layer, buffer) -> {
          if (null == buffer.getDelta()) return;
          final double[] currentDelta = currentDirection.get(layer, buffer.target).getDelta();
          for (int i = 0; i < buffer.getDelta().length; i++) {
            final double prevValue = buffer.target[i];
            final double newValue = prevValue + buffer.getDelta()[i] * alpha;
            if (sign(prevValue) != 0 && sign(prevValue) != sign(newValue)) {
              currentDelta[i] = 0;
              buffer.target[i] = 0;
            }
            else {
              buffer.target[i] = newValue;
            }
          }
        });
        final PointSample measure = subject.measure(monitor).setRate(alpha);
        return new LineSearchPoint(measure, currentDirection.dot(measure.delta));
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
  protected int sign(final double weight) {
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
}
