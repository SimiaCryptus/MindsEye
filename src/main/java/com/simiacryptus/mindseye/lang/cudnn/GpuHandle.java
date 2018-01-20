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

import com.simiacryptus.util.lang.StaticResourcePool;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnHandle;
import jcuda.runtime.cudaDeviceProp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * The type Gpu handle.
 */
public class GpuHandle {
  private static final Logger logger = LoggerFactory.getLogger(GpuHandle.class);
  private static final ThreadLocal<GpuHandle> threadContext = new ThreadLocal<>();
  private static final boolean DISABLE = Boolean.parseBoolean(System.getProperty("DISABLE_CUDNN", "false"));
  private static final boolean FORCE_SINGLE_GPU = Boolean.parseBoolean(System.getProperty("FORCE_SINGLE_GPU", "false"));
  /**
   * The constant gpuContexts.
   */
  public static final StaticResourcePool<GpuHandle> POOL = new StaticResourcePool<>(loadGpuContexts());
  private final jcuda.jcudnn.cudnnHandle handle;
  private final String deviceName;
  private final int deviceNumber;
  private volatile cudaDeviceProp deviceProperties;
  
  /**
   * Instantiates a new Cu dnn.
   *
   * @param deviceNumber the device number
   */
  private GpuHandle(final int deviceNumber) {
    this.deviceNumber = deviceNumber;
    if (0 <= this.deviceNumber) {
      handle = new cudnnHandle();
      initThread();
      deviceName = CuDNN.getDeviceName(deviceNumber);
      JCudnn.cudnnCreate(getHandle());
    }
    else {
      handle = null;
      deviceName = null;
    }
    //cudaSetDevice();
  }
  
  /**
   * Run.
   *
   * @param fn the fn
   */
  public static void apply(final Consumer<GpuHandle> fn) {
    GpuHandle threadlocal = threadContext.get();
    if (threadlocal != null) {
      try {
        threadlocal.initThread();
        fn.accept(threadlocal);
      } catch (final Exception e) {
        throw new RuntimeException(e);
      }
    }
    else {
      POOL.apply(exe -> {
        try {
          threadContext.set(exe);
          exe.initThread();
          fn.accept(exe);
        } catch (final Exception e) {
          throw new RuntimeException(e);
        } finally {
          threadContext.remove();
        }
      });
    }
  }
  
  /**
   * Call t.
   *
   * @param <T> the type parameter
   * @param fn  the fn
   * @return the t
   */
  public static <T> T run(final Function<GpuHandle, T> fn) {
    if (POOL.getAll().isEmpty()) {
      return fn.apply(new GpuHandle(-1));
    }
    else {
      GpuHandle threadlocal = threadContext.get();
      if (threadlocal != null) {
        try {
          threadlocal.initThread();
          return fn.apply(threadlocal);
        } catch (final Exception e) {
          throw new RuntimeException(e);
        }
      }
      else {
        return POOL.run(exe -> {
          try {
            threadContext.set(exe);
            exe.initThread();
            return fn.apply(exe);
          } catch (final Exception e) {
            throw new RuntimeException(e);
          } finally {
            threadContext.remove();
          }
        });
      }
    }
  }
  
  /**
   * For each.
   *
   * @param fn the fn
   */
  public static void forEach(final Consumer<? super GpuHandle> fn) {
    POOL.getAll().forEach(x -> {
      x.initThread();
      fn.accept(x);
    });
  }
  
  /**
   * Load gpu contexts list. If the property disableCuDnn is set to true, no GPUs will be recognized. This is useful for
   * testing CPU-only compatibility.
   *
   * @return the list
   */
  private static List<GpuHandle> loadGpuContexts() {
    if (DISABLE) {
      logger.warn("Disabled CuDNN");
      return Arrays.asList();
    }
    final int deviceCount;
    if (FORCE_SINGLE_GPU) {
      logger.warn("Forcing Single-GPU Mode");
      deviceCount = 1;
    }
    else {
      deviceCount = CuDNN.deviceCount();
    }
    logger.info(String.format("Found %s devices", deviceCount));
    final List<Integer> devices = new ArrayList<>();
    for (int d = 0; d < deviceCount; d++) {
      int deviceNumber = d;
      //if(device>0) System.err.println(String.format("IGNORING Device %s - %s", device, getDeviceName(device)));
      CuDNN.withDevice(deviceNumber, () -> {
        logger.info(String.format("Device %s - %s", deviceNumber, CuDNN.getDeviceName(deviceNumber)));
        devices.add(deviceNumber);
        //CuDNN.handle(cudaSetDeviceFlags(cudaDeviceScheduleAuto));
        for (DeviceLimits limit : DeviceLimits.values()) {
          logger.info(String.format("Default Limit %s = %s", limit, limit.get()));
        }
        DeviceLimits.HeapSize.set(16 * 1024 * 1024 * 1024);
        DeviceLimits.FifoSize.set(8 * 1024 * 1024);
        for (DeviceLimits limit : DeviceLimits.values()) {
          logger.info(String.format("Configured Limit %s = %s", limit, limit.get()));
        }
      });
    }
    if (System.getProperties().containsKey("gpus")) {
      List<Integer> devices2 = Arrays.stream(System.getProperty("gpus").split(","))
                                     .map(Integer::parseInt).collect(Collectors.toList());
      devices.clear();
      devices.addAll(devices2);
    }
    logger.info(String.format("Found %s devices; using devices %s", deviceCount, devices));
    return devices.stream()
                  .map(i -> {
                    try {
                      return new GpuHandle(i);
                    } catch (Throwable e) {
                      return null;
                    }
                  }).filter(x -> x != null).collect(Collectors.toList());
  }
  
  /**
   * Init thread.
   */
  public void initThread() {
    CuDNN.setDevice(getDeviceNumber());
  }
  
  @Override
  public String toString() {
    return getClass().getSimpleName() + "{" + deviceNumber + "; " + deviceName + "}@" + Long.toHexString(System.identityHashCode(this));
  }
  
  /**
   * Gets device properties.
   *
   * @return the device properties
   */
  public cudaDeviceProp getDeviceProperties() {
    if (null == deviceProperties) {
      synchronized (this) {
        if (null == deviceProperties) {
          deviceProperties = CuDNN.getDeviceProperties(getDeviceNumber());
        }
      }
    }
    return deviceProperties;
  }
  
  /**
   * Gets device number.
   *
   * @return the device number
   */
  public int getDeviceNumber() {
    return deviceNumber;
  }
  
  /**
   * Gets device name.
   *
   * @return the device name
   */
  public String getDeviceName() {
    return new String(getDeviceProperties().name, Charset.forName("ASCII")).trim();
  }
  
  @Override
  public void finalize() throws Throwable {
    final int result = JCudnn.cudnnDestroy(getHandle());
    CuDNN.log("cudnnDestroy", result, getHandle());
    CuDNN.handle(result);
  }
  
  /**
   * The Cudnn handle.
   *
   * @return the handle
   */
  public cudnnHandle getHandle() {
    return handle;
  }
}
