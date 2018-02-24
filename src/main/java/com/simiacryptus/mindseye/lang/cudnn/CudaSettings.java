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

package com.simiacryptus.mindseye.lang.cudnn;

import com.simiacryptus.mindseye.lang.PersistanceMode;
import com.simiacryptus.mindseye.lang.Settings;

/**
 * The type Cuda settings.
 */
public class CudaSettings implements Settings {
  /**
   * The constant INSTANCE.
   */
  public static final CudaSettings INSTANCE = new CudaSettings();
  private final PersistanceMode workspaceCachePersistance;
  private final long maxTotalMemory;
  private final long maxAllocSize;
  private final double maxIoElements;
  private final int convolutionWorkspaceSizeLimit;
  private final boolean disable;
  private final boolean forceSingleGpu;
  private final int streamsPerGpu;
  private final int maxFilterElements;
  
  private CudaSettings() {
    maxTotalMemory = Settings.get("MAX_TOTAL_MEMORY", 8 * CudaMemory.GiB);
    maxAllocSize = Settings.get("MAX_ALLOC_SIZE", Precision.Double.size * (Integer.MAX_VALUE - 1L));
    maxFilterElements = Settings.get("MAX_FILTER_ELEMENTS", 256 * 1024 * 1024);
    maxIoElements = Settings.get("MAX_IO_ELEMENTS", 1024 * 1024);
    convolutionWorkspaceSizeLimit = Settings.get("CONVOLUTION_WORKSPACE_SIZE_LIMIT", 1024 * 1024 * 1024);
    disable = Settings.get("DISABLE_CUDNN", false);
    forceSingleGpu = Settings.get("FORCE_SINGLE_GPU", false);
    streamsPerGpu = Settings.get("STREAMS_PER_GPU", 4);
    workspaceCachePersistance = Settings.get("CONV_CACHE_MODE", PersistanceMode.WEAK);
  }
  
  /**
   * The Max total memory.
   *
   * @return the max total memory
   */
  public long getMaxTotalMemory() {
    return maxTotalMemory;
  }
  
  /**
   * The Max.
   *
   * @return the max alloc size
   */
  public long getMaxAllocSize() {
    return maxAllocSize;
  }
  
  /**
   * The constant MAX_IO_ELEMENTS.
   *
   * @return the max io elements
   */
  public double getMaxIoElements() {
    return maxIoElements;
  }
  
  /**
   * The constant CONVOLUTION_WORKSPACE_SIZE_LIMIT.
   *
   * @return the convolution workspace size limit
   */
  public int getConvolutionWorkspaceSizeLimit() {
    return convolutionWorkspaceSizeLimit;
  }
  
  /**
   * The constant gpuContexts.
   *
   * @return the boolean
   */
  public boolean isDisable() {
    return disable;
  }
  
  /**
   * The constant FORCE_SINGLE_GPU.
   *
   * @return the boolean
   */
  public boolean isForceSingleGpu() {
    return forceSingleGpu;
  }
  
  /**
   * The constant STREAMS_PER_GPU.
   *
   * @return the streams per gpu
   */
  public int getStreamsPerGpu() {
    return streamsPerGpu;
  }
  
  /**
   * Gets max filter elements.
   *
   * @return the max filter elements
   */
  public int getMaxFilterElements() {
    return maxFilterElements;
  }
  
  /**
   * Gets workspace cache persistance.
   *
   * @return the workspace cache persistance
   */
  public PersistanceMode getWorkspaceCachePersistance() {
    return workspaceCachePersistance;
  }
}
