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

package com.simiacryptus.mindseye.test.data;

import com.simiacryptus.lang.SupplierWeakCache;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.DataLoader;
import com.simiacryptus.util.test.LabeledObject;
import org.apache.commons.io.IOUtils;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.util.List;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Caltech 101 Images When using, please cite: L. Fei-Fei, R. Fergus and P. Perona. Learning generative visual models
 * from few training examples: an incremental Bayesian approach tested on 101 object categories. IEEE. CVPR 2004,
 * Workshop on Generative-Model Based Vision. 2004 For more information see http://www.vision.caltech.edu/Image_Datasets/Caltech101/
 */
public class Caltech101 {

  @Nullable
  private static final DataLoader<LabeledObject<SupplierWeakCache<BufferedImage>>> training = new DataLoader<LabeledObject<SupplierWeakCache<BufferedImage>>>() {
    @Override
    protected void read(@Nonnull final List<LabeledObject<SupplierWeakCache<BufferedImage>>> queue) {
      try {
        @Nullable InputStream stream = null;
        try {
          // Repackaging as a zip is needed - the tar format classes dont work here
          stream = Util.cacheStream(TestUtil.S3_ROOT.resolve("101_ObjectCategories.zip"));
        } catch (@Nonnull NoSuchAlgorithmException | KeyManagementException e) {
          throw new RuntimeException(e);
        }
        final boolean continueLoop = true;
        @Nullable final ZipInputStream tar = new ZipInputStream(stream);
        while (continueLoop) {
          if (Thread.interrupted()) {
            break;
          }
          final ZipEntry entry = tar.getNextEntry();
          if (null == entry) {
            System.err.println("Null Entry");
            break;
          }
          if (0 == entry.getSize()) {
            continue;
          }
          final String category = entry.getName().split("/")[1];
          //System.err.println(String.format("%s -> %s (%s)", entry.getName(), category, entry.getImageSize()));
          final byte[] data = IOUtils.toByteArray(tar, entry.getSize());
          if (!entry.getName().toLowerCase().endsWith(".jpg")) {
            continue;
          }
          queue.add(new LabeledObject<>(new SupplierWeakCache<>(() -> {
            try {
              return ImageIO.read(new ByteArrayInputStream(data));
            } catch (@Nonnull final IOException e) {
              throw new RuntimeException(e);
            }
          }), category));
        }
      } catch (@Nonnull final IOException e) {
        e.printStackTrace();
        throw new RuntimeException(e);
      }
    }
  };

  /**
   * Halt.
   */
  public static void halt() {
    Caltech101.training.stop();
  }

  /**
   * Training data stream stream.
   *
   * @return the stream
   */
  public static Stream<LabeledObject<SupplierWeakCache<BufferedImage>>> trainingDataStream() {
    return Caltech101.training.stream();
  }


}
