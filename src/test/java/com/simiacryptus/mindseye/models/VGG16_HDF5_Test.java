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

package com.simiacryptus.mindseye.models;

import com.simiacryptus.mindseye.layers.cudnn.CuDNN;
import com.simiacryptus.util.io.NotebookOutput;

import java.io.PrintStream;

/**
 * The Keras Zoo contains a deep CNN called VGG16 which is designed to classify images. Import it from an HDF5 file.
 */
public class VGG16_HDF5_Test extends ImageClassifierTestBase {
  
  @Override
  public ImageClassifier getImageClassifier(NotebookOutput log) {
    PrintStream apiLog = new PrintStream(log.file("cuda.log"));
    CuDNN.apiLog.add(apiLog);
    log.p(log.file((String) null, "cuda.log", "GPU Log"));
    
    return log.code(() -> {
      VGG16 vgg16_hdf5 = VGG16.fromS3_HDF5();
      ((HasHDF5) vgg16_hdf5).getHDF5().print();
      return vgg16_hdf5;
    });
  }
  
  @Override
  protected Class<?> getTargetClass() {
    return VGG16_HDF5.class;
  }
  
}
