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

package com.simiacryptus.mindseye.app;

import com.simiacryptus.mindseye.layers.cudnn.BandReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.models.Hdf5Archive;
import com.simiacryptus.mindseye.models.ImageClassifier;
import com.simiacryptus.mindseye.models.VGG16;
import com.simiacryptus.mindseye.models.VGG16_HDF5;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.Util;

import javax.annotation.Nonnull;

/**
 * The type Image classifier apply base.
 */
public class ObjectLocation_VGG16 extends ObjectLocationBase {
  
  /**
   * Gets target class.
   *
   * @return the target class
   */
  @Nonnull
  protected Class<?> getTargetClass() {
    return VGG16.class;
  }
  
  
  /**
   * The Texture netork.
   */

  @Override
  public ImageClassifier getLocatorNetwork() {
    ImageClassifier locator;
    try {
      locator = new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
        @Override
        protected void phase3b() {
          add(new BandReducerLayer().setMode(getFinalPoolingMode()));
        }
      }//.setSamples(5).setDensity(0.3)
        .setFinalPoolingMode(PoolingLayer.PoolingMode.Avg);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    return locator;
  }
  
  @Override
  public ImageClassifier getClassifierNetwork() {
    ImageClassifier classifier;
    try {
      classifier = new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
        @Override
        protected void phase3b() {
          add(new SoftmaxActivationLayer()
            .setAlgorithm(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE)
            .setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL));
          add(new BandReducerLayer().setMode(getFinalPoolingMode()));
        }
      }//.setSamples(5).setDensity(0.3)
        .setFinalPoolingMode(PoolingLayer.PoolingMode.Max);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    return classifier;
  }
  
}
