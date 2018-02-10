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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.BinaryChunkIterator;
import com.simiacryptus.util.io.DataLoader;
import com.simiacryptus.util.test.LabeledObject;

import javax.annotation.Nullable;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.security.KeyManagementException;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.util.*;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import java.util.zip.GZIPInputStream;

/**
 * References: [LeCun et al., 1998a] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to
 * document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998. See Also:
 * http://yann.lecun.com/exdb/mnist/
 */
public class MNIST {
  
  /**
   * The constant training.
   */
  public static final DataLoader<LabeledObject<Tensor>> training = new DataLoader<LabeledObject<Tensor>>() {
    @Override
    protected void read(@javax.annotation.Nonnull final List<LabeledObject<Tensor>> queue) {
      try {
        final Stream<Tensor> imgStream = MNIST.binaryStream("train-images-idx3-ubyte.gz", 16, 28 * 28).map(b -> {
          return MNIST.fillImage(b, new Tensor(28, 28, 1));
        });
        @javax.annotation.Nonnull final Stream<byte[]> labelStream = MNIST.binaryStream("train-labels-idx1-ubyte.gz", 8, 1);
  
        @javax.annotation.Nonnull final Stream<LabeledObject<Tensor>> merged = MNIST.toStream(new Iterator<LabeledObject<Tensor>>() {
          @javax.annotation.Nonnull
          Iterator<Tensor> imgItr = imgStream.iterator();
          @javax.annotation.Nonnull
          Iterator<byte[]> labelItr = labelStream.iterator();
          
          @Override
          public boolean hasNext() {
            return imgItr.hasNext() && labelItr.hasNext();
          }
    
          @javax.annotation.Nonnull
          @Override
          public LabeledObject<Tensor> next() {
            return new LabeledObject<>(imgItr.next(), Arrays.toString(labelItr.next()));
          }
        }, 100);
        merged.forEach(x -> queue.add(x));
      } catch (@javax.annotation.Nonnull final IOException e) {
        throw new RuntimeException(e);
      }
    }
  };
  /**
   * The constant validation.
   */
  public static final DataLoader<LabeledObject<Tensor>> validation = new DataLoader<LabeledObject<Tensor>>() {
    @Override
    protected void read(@javax.annotation.Nonnull final List<LabeledObject<Tensor>> queue) {
      try {
        final Stream<Tensor> imgStream = MNIST.binaryStream("t10k-images-idx3-ubyte.gz", 16, 28 * 28).map(b -> {
          return MNIST.fillImage(b, new Tensor(28, 28, 1));
        });
        @javax.annotation.Nonnull final Stream<byte[]> labelStream = MNIST.binaryStream("t10k-labels-idx1-ubyte.gz", 8, 1);
  
        @javax.annotation.Nonnull final Stream<LabeledObject<Tensor>> merged = MNIST.toStream(new Iterator<LabeledObject<Tensor>>() {
          @javax.annotation.Nonnull
          Iterator<Tensor> imgItr = imgStream.iterator();
          @javax.annotation.Nonnull
          Iterator<byte[]> labelItr = labelStream.iterator();
          
          @Override
          public boolean hasNext() {
            return imgItr.hasNext() && labelItr.hasNext();
          }
    
          @javax.annotation.Nonnull
          @Override
          public LabeledObject<Tensor> next() {
            return new LabeledObject<>(imgItr.next(), Arrays.toString(labelItr.next()));
          }
        }, 100);
        merged.forEach(x -> queue.add(x));
      } catch (@javax.annotation.Nonnull final IOException e) {
        throw new RuntimeException(e);
      }
    }
  };
  
  private static Stream<byte[]> binaryStream(@javax.annotation.Nonnull final String name, final int skip, final int recordSize) throws IOException {
    @Nullable InputStream stream = null;
    try {
      stream = Util.cacheStream(TestUtil.S3_ROOT.resolve(name));
    } catch (@javax.annotation.Nonnull NoSuchAlgorithmException | KeyStoreException | KeyManagementException e) {
      throw new RuntimeException(e);
    }
    final byte[] fileData = org.apache.commons.io.IOUtils.toByteArray(new java.io.BufferedInputStream(new GZIPInputStream(new java.io.BufferedInputStream(stream))));
    @javax.annotation.Nonnull final DataInputStream in = new DataInputStream(new java.io.ByteArrayInputStream(fileData));
    in.skip(skip);
    return MNIST.toIterator(new BinaryChunkIterator(in, recordSize));
  }
  
  @javax.annotation.Nonnull
  private static Tensor fillImage(final byte[] b, @javax.annotation.Nonnull final Tensor tensor) {
    for (int x = 0; x < 28; x++) {
      for (int y = 0; y < 28; y++) {
        tensor.set(new int[]{x, y}, b[x + y * 28] & 0xFF);
      }
    }
    return tensor;
  }
  
  private static <T> Stream<T> toIterator(@javax.annotation.Nonnull final Iterator<T> iterator) {
    return StreamSupport.stream(Spliterators.spliterator(iterator, 1, Spliterator.ORDERED), false);
  }
  
  private static <T> Stream<T> toStream(@javax.annotation.Nonnull final Iterator<T> iterator, final int size) {
    return MNIST.toStream(iterator, size, false);
  }
  
  private static <T> Stream<T> toStream(@javax.annotation.Nonnull final Iterator<T> iterator, final int size, final boolean parallel) {
    return StreamSupport.stream(Spliterators.spliterator(iterator, size, Spliterator.ORDERED), parallel);
  }
  
  /**
   * Training data stream stream.
   *
   * @return the stream
   * @throws IOException the io exception
   */
  public static Stream<LabeledObject<Tensor>> trainingDataStream() throws IOException {
    return MNIST.training.stream();
  }
  
  /**
   * Validation data stream stream.
   *
   * @return the stream
   * @throws IOException the io exception
   */
  public static Stream<LabeledObject<Tensor>> validationDataStream() throws IOException {
    return MNIST.validation.stream();
  }
  
}
