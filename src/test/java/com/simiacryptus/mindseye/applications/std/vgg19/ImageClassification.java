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

package com.simiacryptus.mindseye.applications.std.vgg19;

import com.simiacryptus.mindseye.applications.ImageClassificationBase;
import com.simiacryptus.mindseye.models.ImageClassifier;
import com.simiacryptus.mindseye.models.VGG19;
import com.simiacryptus.util.io.NotebookOutput;

import javax.annotation.Nonnull;

/**
 * We load a pretrained convolutional neural network (VGG16) along apply the CalTech101 png dataset to perform a
 * demonstration of Image Recognition.
 */
public abstract class ImageClassification extends ImageClassificationBase {
  
  /**
   * Gets target class.
   *
   * @return the target class
   */
  @Nonnull
  protected Class<?> getTargetClass() {
    return VGG19.class;
  }
  
  /**
   * The type HDF5_JBLAS.
   */
  public static class HFD5 extends ImageClassification {
    
    /**
     * Load model png classifier.
     *
     * @param log the log
     * @return the png classifier
     */
    @Override
    public ImageClassifier loadModel(@Nonnull final NotebookOutput log) {
      return log.eval(() -> {
        ImageClassifier classifier = VGG19.fromHDF5();
        classifier.getNetwork();
        return classifier;
      });
    }
    
  }
  
}
