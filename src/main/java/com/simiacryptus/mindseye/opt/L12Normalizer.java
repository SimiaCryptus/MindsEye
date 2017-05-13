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

public class L12Normalizer implements Trainable {
  public final Trainable inner;
  private double factor_L1 = 0.0001;
  private double factor_L2 = 0.0;
  
  public L12Normalizer(Trainable inner) {
    this.inner = inner;
  }
  
  @Override
  public PointSample measure() {
    PointSample innerMeasure = inner.measure();
    DeltaSet normalizationVector = new DeltaSet();
    for (NNLayer layer : getLayers(innerMeasure.delta.map.keySet())) {
      double[] weights = innerMeasure.delta.map.get(layer).target;
      double[] delta = normalizationVector.get(layer, weights).delta;
      for (int i = 0; i < delta.length; i++) {
        delta[i] += factor_L1 * (weights[i] < 0 ? -1.0 : 1.0) + factor_L2 * weights[i];
      }
      assert (null != delta);
    }
    return new PointSample(
                              innerMeasure.delta.add(normalizationVector),
                              innerMeasure.weights,
                              innerMeasure.value);
  }
  
  @Override
  public void resetToFull() {
    inner.resetToFull();
  }
  
  @Override
  public void resetSampling() {
    inner.resetSampling();
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
  
  public L12Normalizer setFactor_L1(double factor_L1) {
    this.factor_L1 = factor_L1;
    return this;
  }
  
  public double getFactor_L2() {
    return factor_L2;
  }
  
  public L12Normalizer setFactor_L2(double factor_L2) {
    this.factor_L2 = factor_L2;
    return this;
  }
}
