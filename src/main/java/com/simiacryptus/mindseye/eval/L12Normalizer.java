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

package com.simiacryptus.mindseye.eval;

import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import org.jetbrains.annotations.NotNull;

import java.util.Collection;
import java.util.stream.Collectors;

/**
 * Abstract base class for a trainable wrapper that adds per-layer L1 and L2 normalization constants. It allows the
 * implementing class to choose the coefficients for each layer.
 */
public abstract class L12Normalizer extends TrainableBase {
  /**
   * The Inner.
   */
  public final Trainable inner;
  private final boolean hideAdj = false;
  
  /**
   * Instantiates a new L 12 normalizer.
   *
   * @param inner the heapCopy
   */
  public L12Normalizer(final Trainable inner) {
    this.inner = inner;
    this.inner.addRef();
  }
  
  @Override
  protected void _free() {
    this.inner.freeRef();
  }
  
  /**
   * Gets l 1.
   *
   * @param layer the layer
   * @return the l 1
   */
  protected abstract double getL1(NNLayer layer);
  
  /**
   * Gets l 2.
   *
   * @param layer the layer
   * @return the l 2
   */
  protected abstract double getL2(NNLayer layer);
  
  /**
   * Gets layers.
   *
   * @param layers the layers
   * @return the layers
   */
  public Collection<NNLayer> getLayers(final @NotNull Collection<NNLayer> layers) {
    return layers.stream()
                 .filter(layer -> {
                   return layer instanceof FullyConnectedLayer;
                 })
                 .collect(Collectors.toList());
  }
  
  
  @Override
  public PointSample measure(final TrainingMonitor monitor) {
    final PointSample innerMeasure = inner.measure(monitor);
    final @NotNull DeltaSet<NNLayer> normalizationVector = new DeltaSet<NNLayer>();
    double valueAdj = 0;
    for (final @NotNull NNLayer layer : getLayers(innerMeasure.delta.getMap().keySet())) {
      final double[] weights = innerMeasure.delta.getMap().get(layer).target;
      final double[] gradientAdj = normalizationVector.get(layer, weights).getDelta();
      final double factor_L1 = getL1(layer);
      final double factor_L2 = getL2(layer);
      for (int i = 0; i < gradientAdj.length; i++) {
        final double sign = weights[i] < 0 ? -1.0 : 1.0;
        gradientAdj[i] += factor_L1 * sign + 2 * factor_L2 * weights[i];
        valueAdj += (factor_L1 * sign + factor_L2 * weights[i]) * weights[i];
      }
      assert null != gradientAdj;
    }
    return new PointSample(
      innerMeasure.delta.add(normalizationVector),
      innerMeasure.weights,
      innerMeasure.sum + (hideAdj ? 0 : valueAdj),
      innerMeasure.rate,
      innerMeasure.count).normalize();
  }
  
  @Override
  public boolean reseed(final long seed) {
    return inner.reseed(seed);
  }
  
}
