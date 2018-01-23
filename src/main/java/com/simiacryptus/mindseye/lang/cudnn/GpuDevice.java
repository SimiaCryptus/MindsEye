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

import jcuda.runtime.cudaDeviceProp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.Charset;

public class GpuDevice {
  protected static final Logger logger = LoggerFactory.getLogger(GpuHandle.class);
  protected final String deviceName;
  protected final int deviceNumber;
  private volatile cudaDeviceProp deviceProperties;
  
  public GpuDevice(final int deviceNumber) {
    this.deviceNumber = deviceNumber;
    if (0 <= this.deviceNumber) {
      initThread();
      deviceName = CuDNN.getDeviceName(deviceNumber);
    }
    else {
      deviceName = null;
    }
  }
  
  /**
   * Init thread.
   */
  public void initThread() {
    CuDNN.setDevice(getDeviceNumber());
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
}
