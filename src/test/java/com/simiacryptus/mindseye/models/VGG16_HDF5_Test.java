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

package com.simiacryptus.mindseye.models;

import com.simiacryptus.util.io.NotebookOutput;

/**
 * The type Layer run base.
 */
public class VGG16_HDF5_Test extends ImageClassifierTestBase {
  
  @Override
  public ImageClassifier getImageClassifier(NotebookOutput log) {
    return log.code(() -> {
      VGG16_HDF5 vgg16_hdf5 = VGG16_HDF5.fromS3();
      vgg16_hdf5.getHDF5().print();
      return vgg16_hdf5;
    });
  }
  
  @Override
  protected Class<?> getTargetClass() {
    return VGG16_HDF5.class;
  }
  
}
