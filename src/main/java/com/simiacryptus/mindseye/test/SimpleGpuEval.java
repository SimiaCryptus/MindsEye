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

package com.simiacryptus.mindseye.test;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensorList;
import com.simiacryptus.mindseye.lang.cudnn.CudnnHandle;
import com.simiacryptus.mindseye.lang.cudnn.MemoryType;
import com.simiacryptus.mindseye.lang.cudnn.Precision;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * The type Simple gpu eval.
 */
public class SimpleGpuEval extends SimpleListEval {
  
  private final CudnnHandle gpu;
  
  /**
   * Instantiates a new Simple gpu eval.
   *
   * @param layer the layer
   * @param gpu   the gpu
   * @param input the input
   */
  public SimpleGpuEval(@Nonnull Layer layer, CudnnHandle gpu, TensorList... input) {
    super(layer, input);
    this.gpu = gpu;
  }
  
  /**
   * Run simple result.
   *
   * @param layer  the layer
   * @param gpu    the gpu
   * @param tensor the tensor
   * @return the simple result
   */
  public static SimpleResult run(@Nonnull final Layer layer, final CudnnHandle gpu, final TensorList... tensor) {
    return new SimpleGpuEval(layer, gpu, tensor).call();
  }
  
  @Nonnull
  @Override
  public TensorList getFeedback(@Nonnull final TensorList original) {
    return toGpu(getDelta(original));
  }
  
  /**
   * To gpu cuda tensor list.
   *
   * @param tensorArray the tensor array
   * @return the cuda tensor list
   */
  @Nonnull
  public CudaTensorList toGpu(final TensorArray tensorArray) {
    @Nullable CudaTensor cudaMemory = gpu.getTensor(tensorArray, Precision.Double, MemoryType.Managed.normalize(), false);
    tensorArray.freeRef();
    return CudaTensorList.wrap(cudaMemory, tensorArray.length(), tensorArray.getDimensions(), Precision.Double);
  }
  
  /**
   * Gets delta.
   *
   * @param output the output
   * @return the delta
   */
  @Nonnull
  public TensorArray getDelta(final TensorList output) {
    return TensorArray.wrap(output.stream().map(t -> {
      @Nullable Tensor map = t.map(v -> 1.0);
      t.freeRef();
      return map;
    }).toArray(i -> new Tensor[i]));
  }
  
}
