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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.ImgBandBiasLayer;

/**
 * This Trainable wrapper adds additional L1 and L2 terms for weight normalization. Both coefficients are universal for
 * the network (does not depend on layer) and are setByCoord statically.
 */
public class ConstL12Normalizer extends L12Normalizer implements SampledTrainable, TrainableDataMask {
  private double factor_L1 = 0.0;
  private double factor_L2 = 0.0;
  
  /**
   * Instantiates a new Const l 12 normalizer.
   *
   * @param inner the inner
   */
  public ConstL12Normalizer(final Trainable inner) {
    super(inner);
  }
  
  @Override
  public SampledCachedTrainable<? extends SampledTrainable> cached() {
    return new SampledCachedTrainable<>(this);
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
  public ConstL12Normalizer setFactor_L1(final double factor_L1) {
    this.factor_L1 = factor_L1;
    return this;
  }
  
  /**
   * Gets factor l 2.
   *
   * @return the factor l 2
   */
  public double getFactor_L2() {
    return factor_L2;
  }
  
  /**
   * Sets factor l 2.
   *
   * @param factor_L2 the factor l 2
   * @return the factor l 2
   */
  public ConstL12Normalizer setFactor_L2(final double factor_L2) {
    this.factor_L2 = factor_L2;
    return this;
  }
  
  @Override
  protected double getL1(final NNLayer layer) {
    if (supress(layer)) return 0;
    return factor_L1;
  }
  
  @Override
  protected double getL2(final NNLayer layer) {
    return factor_L2;
  }
  
  @Override
  public boolean[] getMask() {
    return ((TrainableDataMask) inner).getMask();
  }
  
  @Override
  public int getTrainingSize() {
    return ((SampledTrainable) inner).getTrainingSize();
  }
  
  @Override
  public ConstL12Normalizer setTrainingSize(final int trainingSize) {
    ((SampledTrainable) inner).setTrainingSize(trainingSize);
    return this;
  }
  
  @Override
  public TrainableDataMask setMask(final boolean... mask) {
    ((TrainableDataMask) inner).setMask(mask);
    return this;
  }
  
  private boolean supress(final NNLayer layer) {
    if (layer instanceof BiasLayer) return false;
    if (layer instanceof ImgBandBiasLayer) return false;
    return false;
  }
}
