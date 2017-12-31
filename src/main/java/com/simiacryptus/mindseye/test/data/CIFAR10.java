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

package com.simiacryptus.mindseye.test.data;

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.BinaryChunkIterator;
import com.simiacryptus.util.io.DataLoader;
import com.simiacryptus.util.test.LabeledObject;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.io.input.BoundedInputStream;

import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.security.KeyManagementException;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

/**
 * Mirrored from https://www.cs.toronto.edu/~kriz/cifar.html For more information, and for citation, please see:
 * Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009. https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
 */
public class CIFAR10 {
  
  private static final DataLoader<LabeledObject<Tensor>> training = new DataLoader<LabeledObject<Tensor>>() {
    @Override
    protected void read(final List<LabeledObject<Tensor>> queue) {
      try {
        InputStream stream = null;
        try {
          stream = Util.cacheStream(TestUtil.S3_ROOT.resolve("cifar-10-binary.tar.gz"));
        } catch (NoSuchAlgorithmException | KeyStoreException | KeyManagementException e) {
          throw new RuntimeException(e);
        }
        final int recordSize = 3073;
        final GZIPInputStream inflatedInput = new GZIPInputStream(stream);
        final TarArchiveInputStream tar = new TarArchiveInputStream(inflatedInput);
        while (0 < inflatedInput.available()) {
          if (Thread.interrupted()) {
            break;
          }
          final TarArchiveEntry nextTarEntry = tar.getNextTarEntry();
          if (null == nextTarEntry) {
            break;
          }
          final BinaryChunkIterator iterator = new BinaryChunkIterator(new DataInputStream(new BoundedInputStream(tar, nextTarEntry.getSize())), recordSize);
          for (final byte[] chunk : (Iterable<byte[]>) () -> iterator) {
            queue.add(CIFAR10.toImage(chunk).map(img -> Tensor.fromRGB(img)));
          }
        }
        System.err.println("Done loading");
      } catch (final IOException e) {
        e.printStackTrace();
        throw new RuntimeException(e);
      }
    }
  };
  
  /**
   * Halt.
   */
  public static void halt() {
    CIFAR10.training.stop();
  }
  
  private static LabeledObject<BufferedImage> toImage(final byte[] b) {
    final BufferedImage img = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
    for (int x = 0; x < img.getWidth(); x++) {
      for (int y = 0; y < img.getHeight(); y++) {
        final int red = 0xFF & b[1 + 1024 * 0 + x + y * 32];
        final int blue = 0xFF & b[1 + 1024 * 1 + x + y * 32];
        final int green = 0xFF & b[1 + 1024 * 2 + x + y * 32];
        final int c = (red << 16) + (blue << 8) + green;
        img.setRGB(x, y, c);
      }
    }
    return new LabeledObject<>(img, Arrays.toString(new byte[]{b[0]}));
  }
  
  /**
   * Training data stream stream.
   *
   * @return the stream
   * @throws IOException the io exception
   */
  public static Stream<LabeledObject<Tensor>> trainingDataStream() throws IOException {
    return CIFAR10.training.stream();
  }
  
  
}
