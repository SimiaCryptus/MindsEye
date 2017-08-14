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

package com.simiacryptus.mindseye.layers.cudnn.f32;

import com.simiacryptus.mindseye.layers.cudnn.CuDNN;
import com.simiacryptus.mindseye.layers.cudnn.CudaPtr;
import com.simiacryptus.util.ml.Tensor;

import java.util.Random;
import java.util.stream.IntStream;

public class GPUDataMirror {
  private long fingerprint;
  private int[] indicies;
  private volatile CudaPtr ptr;

  public GPUDataMirror(int length) {
    this.indicies = IntStream.range(0, 3).map(i -> new Random().nextInt(length)).distinct().limit(3).toArray();
  }

  public CudaPtr upload(int device, float[] data) {
    long inputHash = hashFunction(data);
    if(null != ptr && inputHash == fingerprint) return ptr;
    this.fingerprint = inputHash;
    return ptr = CuDNN.write(device, data);
  }

  public CudaPtr upload(int device, double[] data) {
    long inputHash = hashFunction(data);
    if(null != ptr && inputHash == fingerprint) return ptr;
    this.fingerprint = inputHash;
    return ptr = CuDNN.write(device, data);
  }

  public CudaPtr uploadAsFloats(int device, double[] data) {
    long inputHash = hashFunction(data);
    if(null != ptr && inputHash == fingerprint) return ptr;
    this.fingerprint = inputHash;
    return ptr = CuDNN.write(device, Tensor.toFloats(data));
  }

  public long hashFunction(float[] data) {
    return IntStream.of(indicies).mapToObj(i->data[i])
             .mapToInt(Float::floatToIntBits)
             .reduce((a,b)->a^b).getAsInt();
  }
  public long hashFunction(double[] data) {
    return IntStream.of(indicies).mapToDouble(i->data[i])
             .mapToLong(Double::doubleToLongBits)
             .reduce((a,b)->a^b).getAsLong();
  }
}
