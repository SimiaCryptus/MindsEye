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

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.nio.charset.Charset;

/**
 * The type Gpu device.
 */
public class GpuDevice extends GpuSystem {
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(CuDNNHandle.class);
  /**
   * The Device name.
   */
  @Nullable
  protected final String deviceName;
  /**
   * The Device number.
   */
  protected final int deviceNumber;
  private volatile cudaDeviceProp deviceProperties;
  
  /**
   * Instantiates a new Gpu device.
   *
   * @param deviceNumber the device number
   */
  public GpuDevice(final int deviceNumber) {
    super();
    this.deviceNumber = deviceNumber;
    if (0 <= this.deviceNumber) {
      initThread();
      deviceName = getDeviceName(deviceNumber);
    }
    else {
      deviceName = null;
    }
  }
  
  /**
   * Cuda freeRef int.
   *
   * @param devPtr   the dev ptr
   * @param deviceId the device id
   * @return the int
   */
  public static int cudaFree(final Pointer devPtr, int deviceId) {
    long startTime = System.nanoTime();
    return GpuSystem.withDevice(deviceId, () -> {
      final int result = JCuda.cudaFree(devPtr);
      GpuSystem.log("cudaFree", result, devPtr);
      cudaFree_execution.accept((System.nanoTime() - startTime) / 1e9);
      handle(result);
      return result;
    });
  }
  
  /**
   * Gets device name.
   *
   * @param device the device
   * @return the device name
   */
  public static String getDeviceName(final int device) {
    return new String(GpuDevice.getDeviceProperties(device).name, Charset.forName("ASCII")).trim();
  }
  
  /**
   * Gets device properties.
   *
   * @param device the device
   * @return the device properties
   */
  public static cudaDeviceProp getDeviceProperties(final int device) {
    return propertyCache.computeIfAbsent(device, deviceId -> {
      long startTime = System.nanoTime();
      @javax.annotation.Nonnull final cudaDeviceProp deviceProp = new cudaDeviceProp();
      final int result = JCuda.cudaGetDeviceProperties(deviceProp, device);
      getDeviceProperties_execution.accept((System.nanoTime() - startTime) / 1e9);
      GpuSystem.log("cudaGetDeviceProperties", result, deviceProp, device);
      return deviceProp;
    });
  }
  
  /**
   * Sets device.
   *
   * @param cudaDeviceId the cuda device id
   */
  public static void setDevice(final int cudaDeviceId) {
    if (cudaDeviceId < 0) throw new IllegalArgumentException("cudaDeviceId=" + cudaDeviceId);
    if (cudaDeviceId != getDevice()) {
      long startTime = System.nanoTime();
      final int result = JCuda.cudaSetDevice(cudaDeviceId);
      setDevice_execution.accept((System.nanoTime() - startTime) / 1e9);
      GpuSystem.log("cudaSetDevice", result, cudaDeviceId);
      GpuSystem.handle(result);
      GpuSystem.currentDevice.set(cudaDeviceId);
    }
  }
  
  /**
   * Init thread.
   */
  public void initThread() {
    setDevice(getDeviceNumber());
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
          deviceProperties = getDeviceProperties(getDeviceNumber());
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
  @javax.annotation.Nonnull
  public String getDeviceName() {
    return new String(getDeviceProperties().name, Charset.forName("ASCII")).trim();
  }
}
