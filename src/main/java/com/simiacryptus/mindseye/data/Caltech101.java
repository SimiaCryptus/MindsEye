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

package com.simiacryptus.mindseye.data;

import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.DataLoader;
import com.simiacryptus.util.lang.SupplierWeakCache;
import com.simiacryptus.util.test.LabeledObject;
import org.apache.commons.io.IOUtils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.security.KeyManagementException;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.util.List;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * The type Caltech 101.
 */
public class Caltech101 {
  
  private static final URI source = URI.create("https://s3-us-west-2.amazonaws.com/simiacryptus/");
  
  private static final DataLoader training = new DataLoader<LabeledObject<SupplierWeakCache<BufferedImage>>>() {
    @Override
    protected void read(List<LabeledObject<SupplierWeakCache<BufferedImage>>> queue) {
      try {
        InputStream stream = null;
        try {
          // Repackaging as a zip is needed - the tar format classes dont work here
          stream = Util.cache(source.resolve("101_ObjectCategories.zip"));
        } catch (NoSuchAlgorithmException | KeyStoreException | KeyManagementException e) {
          throw new RuntimeException(e);
        }
        boolean continueLoop = true;
        ZipInputStream tar = new ZipInputStream(stream);
        while (continueLoop) {
          if (Thread.interrupted()) break;
          ZipEntry entry = tar.getNextEntry();
          if (null == entry) {
            System.err.println("Null Entry");
            break;
          }
          if (0 == entry.getSize()) continue;
          String category = entry.getName().split("/")[1];
          //System.err.println(String.format("%s -> %s (%s)", entry.getName(), category, entry.getSize()));
          byte[] data = IOUtils.toByteArray(tar, entry.getSize());
          if (!entry.getName().toLowerCase().endsWith(".jpg")) continue;
          queue.add(new LabeledObject<>(new SupplierWeakCache<BufferedImage>(() -> {
            try {
              return ImageIO.read(new ByteArrayInputStream(data));
            } catch (IOException e) {
              throw new RuntimeException(e);
            }
          }), category));
        }
      } catch (IOException e) {
        e.printStackTrace();
        throw new RuntimeException(e);
      }
    }
  };
  
  /**
   * Training data stream stream.
   *
   * @return the stream
   * @throws IOException the io exception
   */
  public static Stream<LabeledObject<SupplierWeakCache<BufferedImage>>> trainingDataStream() throws IOException {
    return training.stream();
  }
  
  /**
   * Halt.
   */
  public static void halt() {
    training.stop();
  }
  
  
}
