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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.BinaryChunkIterator;
import com.simiacryptus.util.io.DataLoader;
import com.simiacryptus.util.test.LabeledObject;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.security.KeyManagementException;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.util.*;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import java.util.zip.GZIPInputStream;

/**
 * The type Mnist.
 */
public class MNIST {
  
  private final static URI source = URI.create("http://yann.lecun.com/exdb/mnist/");
  
  private static Stream<byte[]> binaryStream(final String name, final int skip, final int recordSize) throws IOException {
    InputStream stream = null;
    try {
      stream = Util.cache(source.resolve(name));
    } catch (NoSuchAlgorithmException | KeyStoreException | KeyManagementException e) {
      throw new RuntimeException(e);
    }
    byte[] fileData = org.apache.commons.io.IOUtils.toByteArray(new java.io.BufferedInputStream(new GZIPInputStream(new java.io.BufferedInputStream(stream))));
    final DataInputStream in = new DataInputStream(new java.io.ByteArrayInputStream(fileData));
    in.skip(skip);
    return toIterator(new BinaryChunkIterator(in, recordSize));
  }
  
  private static <T> Stream<T> toIterator(final Iterator<T> iterator) {
    return StreamSupport.stream(Spliterators.spliterator(iterator, 1, Spliterator.ORDERED), false);
  }
  
  /**
   * The constant training.
   */
  public static final DataLoader training = new DataLoader<LabeledObject<Tensor>>() {
    @Override
    protected void read(List<LabeledObject<Tensor>> queue) {
      try {
        final Stream<Tensor> imgStream = binaryStream("train-images-idx3-ubyte.gz", 16, 28 * 28).map(b -> {
          return fillImage(b, new Tensor(28, 28, 1));
        });
        final Stream<byte[]> labelStream = binaryStream("train-labels-idx1-ubyte.gz", 8, 1);
        
        final Stream<LabeledObject<Tensor>> merged = toStream(new Iterator<LabeledObject<Tensor>>() {
          Iterator<Tensor> imgItr = imgStream.iterator();
          Iterator<byte[]> labelItr = labelStream.iterator();
          
          @Override
          public boolean hasNext() {
            return this.imgItr.hasNext() && this.labelItr.hasNext();
          }
          
          @Override
          public LabeledObject<Tensor> next() {
            return new LabeledObject<Tensor>(this.imgItr.next(), Arrays.toString(this.labelItr.next()));
          }
        }, 100);
        merged.forEach(x -> queue.add(x));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  };
  
  /**
   * The constant validation.
   */
  public static final DataLoader validation = new DataLoader<LabeledObject<Tensor>>() {
    @Override
    protected void read(List<LabeledObject<Tensor>> queue) {
      try {
        final Stream<Tensor> imgStream = binaryStream("t10k-images-idx3-ubyte.gz", 16, 28 * 28).map(b -> {
          return fillImage(b, new Tensor(28, 28, 1));
        });
        final Stream<byte[]> labelStream = binaryStream("t10k-labels-idx1-ubyte.gz", 8, 1);
        
        final Stream<LabeledObject<Tensor>> merged = toStream(new Iterator<LabeledObject<Tensor>>() {
          Iterator<Tensor> imgItr = imgStream.iterator();
          Iterator<byte[]> labelItr = labelStream.iterator();
          
          @Override
          public boolean hasNext() {
            return this.imgItr.hasNext() && this.labelItr.hasNext();
          }
          
          @Override
          public LabeledObject<Tensor> next() {
            return new LabeledObject<Tensor>(this.imgItr.next(), Arrays.toString(this.labelItr.next()));
          }
        }, 100);
        merged.forEach(x -> queue.add(x));
      } catch (IOException e) {
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
  public static Stream<LabeledObject<Tensor>> trainingDataStream() throws IOException {
    return training.stream();
  }
  
  private static Tensor fillImage(final byte[] b, final Tensor tensor) {
    for (int x = 0; x < 28; x++) {
      for (int y = 0; y < 28; y++) {
        tensor.set(new int[]{x, y}, b[x + y * 28] & 0xFF);
      }
    }
    return tensor;
  }
  
  private static <T> Stream<T> toStream(final Iterator<T> iterator, final int size) {
    return toStream(iterator, size, false);
  }
  
  private static <T> Stream<T> toStream(final Iterator<T> iterator, final int size, final boolean parallel) {
    return StreamSupport.stream(Spliterators.spliterator(iterator, size, Spliterator.ORDERED), parallel);
  }
  
  /**
   * Validation data stream stream.
   *
   * @return the stream
   * @throws IOException the io exception
   */
  public static Stream<LabeledObject<Tensor>> validationDataStream() throws IOException {
    return validation.stream();
  }
  
}
