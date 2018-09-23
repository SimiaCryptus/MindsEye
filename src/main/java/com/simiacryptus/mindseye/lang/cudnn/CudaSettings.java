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
import com.simiacryptus.util.CodeUtil;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.LocalAppSettings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.HashMap;

/**
 * The type Cuda settings.
 */
public class CudaSettings implements Settings {

  private static final Logger logger = LoggerFactory.getLogger(CudaSettings.class);

  private static transient CudaSettings INSTANCE = null;
  public final String defaultDevices;
  /**
   * The Memory cacheLocal mode.
   */
  public final PersistanceMode memoryCacheMode;
  private final long maxTotalMemory;
  private final long maxAllocSize;
  private final double maxIoElements;
  private final long convolutionWorkspaceSizeLimit;
  private final boolean disable;
  private final boolean forceSingleGpu;
  private final long maxFilterElements;
  private final boolean conv_para_2;
  private final boolean conv_para_1;
  private final boolean conv_para_3;
  private final long maxDeviceMemory;
  private final boolean logStack;
  private final boolean profileMemoryIO;
  private final boolean asyncFree;
  private final boolean enableManaged;
  private final boolean syncBeforeFree;
  private final int memoryCacheTTL;
  private final boolean convolutionCache;

  public static CudaSettings INSTANCE() {
    if(null==INSTANCE) {
      synchronized (CudaSettings.class) {
        if(null==INSTANCE) {
          INSTANCE = new CudaSettings();
          logger.info(String.format("Initialized %s = %s", INSTANCE.getClass().getSimpleName(), JsonUtil.toJson(INSTANCE)));
        }
      }
    }
    return INSTANCE;
  }


  private CudaSettings() {
    HashMap<String, String> appSettings = LocalAppSettings.read();
    String spark_home = System.getenv("SPARK_HOME");
    File sparkHomeFile = new File(spark_home==null?".":spark_home);
    if(sparkHomeFile.exists()) appSettings.putAll(LocalAppSettings.read(sparkHomeFile));
    if(appSettings.containsKey("worker.index")) System.setProperty("CUDA_DEVICES", appSettings.get("worker.index"));
    maxTotalMemory = Settings.get("MAX_TOTAL_MEMORY", 14 * CudaMemory.GiB);
    maxDeviceMemory = Settings.get("MAX_DEVICE_MEMORY", 14 * CudaMemory.GiB);
    maxAllocSize = Settings.get("MAX_ALLOC_SIZE", Precision.Double.size * (Integer.MAX_VALUE - 1L));
    maxFilterElements = Settings.get("MAX_FILTER_ELEMENTS", 1024 * CudaMemory.MiB);
    maxIoElements = Settings.get("MAX_IO_ELEMENTS", 2 * CudaMemory.MiB);
    convolutionWorkspaceSizeLimit = Settings.get("CONVOLUTION_WORKSPACE_SIZE_LIMIT", 512 * CudaMemory.MiB);
    disable = Settings.get("DISABLE_CUDNN", false);
    forceSingleGpu = Settings.get("FORCE_SINGLE_GPU", false);
    conv_para_1 = Settings.get("CONV_PARA_1", false);
    conv_para_2 = Settings.get("CONV_PARA_2", false);
    conv_para_3 = Settings.get("CONV_PARA_3", false);
    memoryCacheMode = Settings.get("CUDA_CACHE_MODE", PersistanceMode.WEAK);
    logStack = Settings.get("CUDA_LOG_STACK", false);
    profileMemoryIO = Settings.get("CUDA_PROFILE_MEM_IO", false);
    enableManaged = true;
    asyncFree = false;
    syncBeforeFree = true;
    memoryCacheTTL = 5;
    convolutionCache = true;
    defaultDevices = Settings.get("CUDA_DEVICES", "");
  }

  /**
   * The Max total memory.
   *
   * @return the max total memory
   */
  public double getMaxTotalMemory() {
    return maxTotalMemory;
  }

  /**
   * The Max.
   *
   * @return the max alloc size
   */
  public double getMaxAllocSize() {
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
  public long getConvolutionWorkspaceSizeLimit() {
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
   * Gets max filter elements.
   *
   * @return the max filter elements
   */
  public long getMaxFilterElements() {
    return maxFilterElements;
  }

  /**
   * Is conv para 2 boolean.
   *
   * @return the boolean
   */
  public boolean isConv_para_2() {
    return conv_para_2;
  }

  /**
   * Is conv para 1 boolean.
   *
   * @return the boolean
   */
  public boolean isConv_para_1() {
    return conv_para_1;
  }

  /**
   * Is conv para 3 boolean.
   *
   * @return the boolean
   */
  public boolean isConv_para_3() {
    return conv_para_3;
  }

  /**
   * Gets max device memory.
   *
   * @return the max device memory
   */
  public double getMaxDeviceMemory() {
    return maxDeviceMemory;
  }

  /**
   * Is log stack boolean.
   *
   * @return the boolean
   */
  public boolean isLogStack() {
    return logStack;
  }

  /**
   * Is profile memory io boolean.
   *
   * @return the boolean
   */
  public boolean isProfileMemoryIO() {
    return profileMemoryIO;
  }

  /**
   * Is async free boolean.
   *
   * @return the boolean
   */
  public boolean isAsyncFree() {
    return asyncFree;
  }

  /**
   * Is enable managed boolean.
   *
   * @return the boolean
   */
  public boolean isEnableManaged() {
    return enableManaged;
  }

  /**
   * The Sync before free.
   *
   * @return the boolean
   */
  public boolean isSyncBeforeFree() {
    return syncBeforeFree;
  }

  /**
   * Gets memory cacheLocal ttl.
   *
   * @return the memory cacheLocal ttl
   */
  public int getMemoryCacheTTL() {
    return memoryCacheTTL;
  }

  /**
   * Is convolution cacheLocal boolean.
   *
   * @return the boolean
   */
  public boolean isConvolutionCache() {
    return convolutionCache;
  }

}
