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

package com.simiacryptus.mindseye.layers.cudnn;

import com.simiacryptus.mindseye.lang.State;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import jcuda.Pointer;
import jcuda.jcudnn.cudnnTensorDescriptor;

import static jcuda.jcudnn.JCudnn.cudnnAddTensor;

/**
 * The type Cudnn float delta.
 */
public class CudnnFloatDelta extends State {
  /**
   * The Buffer.
   */
  CudaPtr buffer;
  
  /**
   * Instantiates a new Cudnn float delta.
   *
   * @param values the values
   * @param layer  the layer
   */
  public CudnnFloatDelta(double[] values, NNLayer layer) {
    super(values, null, layer);
  }
  
  /**
   * Accumulate.
   *
   * @param size  the size
   * @param data  the data
   * @param cudnn the cudnn
   */
  public void accumulate(CudaResource<cudnnTensorDescriptor> size, CudaPtr data, CuDNN cudnn) {
    if (null != buffer) {
      CuDNN.handle(cudnnAddTensor(cudnn.cudnnHandle,
        Pointer.to(new float[]{1.0f}), size.getPtr(), data.getPtr(),
        Pointer.to(new float[]{1.0f}), size.getPtr(), buffer.getPtr()));
      data.finalize();
    }
    else {
      buffer = data;
    }
  }
  
  @Override
  public double[] getDelta() {
    if (null == delta) {
      float[] data = new float[length()];
      buffer.read(data);
      this.delta = Tensor.toDoubles(data);
    }
    return super.getDelta();
  }
}
