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

package com.simiacryptus.mindseye.layers.aparapi;

import com.aparapi.device.Device;
import com.aparapi.internal.kernel.KernelManager;
import com.simiacryptus.util.lang.ResourcePool;

/**
 * The type Open cl.
 */
public final class OpenCL {
  
  /**
   * The constant devicePool.
   */
  public static final ResourcePool<Device> devicePool = new ResourcePool<Device>(Integer.parseInt(System.getProperty("num_gpus", "1"))) {
    @Override
    public Device create() {
      return KernelManager.instance().bestDevice();
    }
  };
}