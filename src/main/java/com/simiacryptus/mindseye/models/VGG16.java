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

import com.simiacryptus.mindseye.applications.ImageClassifierBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.Util;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.IOException;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;

/**
 * Details about this network architecture can be found in the following arXiv paper: Very Deep Convolutional Networks
 * for Large-Scale Image Recognition K. Simonyan, A. Zisserman arXiv:1409.1556 Please cite the paper if you use the
 * models.
 */
public abstract class VGG16 extends VGG {
  
  /**
   * From s 3 vgg 16 hdf 5.
   *
   * @return the vgg 16 hdf 5
   */
  public static ImageClassifierBase fromHDF5() {
    try {
      return fromHDF5(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")));
    } catch (IOException | KeyManagementException | NoSuchAlgorithmException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * From s 3 vgg 16 hdf 5.
   *
   * @param hdf the hdf
   * @return the vgg 16 hdf 5
   */
  public static ImageClassifierBase fromHDF5(final File hdf) {
    try {
      return new VGG16_HDF5(new Hdf5Archive(hdf));
    } catch (@Nonnull final RuntimeException e) {
      throw e;
    } catch (Throwable e) {
      throw new RuntimeException(e);
    }
  }
  
}
