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

import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.media.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.synapse.BiasLayer;

/**
 * Created by Andrew Charneski on 5/26/2017.
 */
public class ConstL12Normalizer extends L12Normalizer {
  private double factor_L1 = 0.0;
  private double factor_L2 = 0.0;
  
  public ConstL12Normalizer(Trainable inner) {
    super(inner);
  }
  
  @Override
  protected double getL1(NNLayer layer) {
    if (supress(layer)) return 0;
    return factor_L1;
  }
  
  private boolean supress(NNLayer layer) {
    if (layer instanceof BiasLayer) return false;
    if (layer instanceof ImgBandBiasLayer) return false;
    return false;
  }
  
  @Override
  protected double getL2(NNLayer layer) {
    return factor_L2;
  }
  
  public double getFactor_L1() {
    return factor_L1;
  }
  
  public com.simiacryptus.mindseye.opt.trainable.ConstL12Normalizer setFactor_L1(double factor_L1) {
    this.factor_L1 = factor_L1;
    return this;
  }
  
  public double getFactor_L2() {
    return factor_L2;
  }
  
  public com.simiacryptus.mindseye.opt.trainable.ConstL12Normalizer setFactor_L2(double factor_L2) {
    this.factor_L2 = factor_L2;
    return this;
  }
}
