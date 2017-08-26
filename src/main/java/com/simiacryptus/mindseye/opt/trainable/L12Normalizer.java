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

package com.simiacryptus.mindseye.opt.trainable;

import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.synapse.JavaDenseSynapseLayer;
import com.simiacryptus.mindseye.layers.synapse.ToeplitzSynapseLayer;

import java.util.Collection;
import java.util.stream.Collectors;

/**
 * The type L 12 normalizer.
 */
public abstract class L12Normalizer implements Trainable {
  /**
   * The Inner.
   */
  public final Trainable inner;
  
  private final boolean hideAdj = false;
  
  /**
   * Instantiates a new L 12 normalizer.
   *
   * @param inner the inner
   */
  public L12Normalizer(Trainable inner) {
    this.inner = inner;
  }
  
  @Override
  public Trainable.PointSample measure() {
    Trainable.PointSample innerMeasure = inner.measure();
    DeltaSet normalizationVector = new DeltaSet();
    double valueAdj = 0;
    for (NNLayer layer : getLayers(innerMeasure.delta.map.keySet())) {
      double[] weights = innerMeasure.delta.map.get(layer).target;
      double[] gradientAdj = normalizationVector.get(layer, weights).getDelta();
      double factor_L1 = getL1(layer);
      double factor_L2 = getL2(layer);
      for (int i = 0; i < gradientAdj.length; i++) {
        double sign = weights[i] < 0 ? -1.0 : 1.0;
        gradientAdj[i] += factor_L1 * sign + 2 * factor_L2 * weights[i];
        valueAdj += (factor_L1 * sign + factor_L2 * weights[i]) * weights[i];
      }
      assert (null != gradientAdj);
    }
    return new Trainable.PointSample(
                              innerMeasure.delta.add(normalizationVector),
                              innerMeasure.weights,
                              innerMeasure.value + (hideAdj?0:valueAdj));
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
  
  @Override
  public void resetToFull() {
    inner.resetToFull();
  }
  
  @Override
  public boolean resetSampling() {
    return inner.resetSampling();
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
  
}
