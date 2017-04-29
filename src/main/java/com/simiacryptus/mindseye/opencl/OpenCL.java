package com.simiacryptus.mindseye.opencl;

import com.aparapi.Kernel;
import com.aparapi.Kernel.EXECUTION_MODE;
import com.aparapi.device.Device.TYPE;
import com.aparapi.internal.kernel.KernelManager;

public final class OpenCL{

  public static Kernel init(final com.aparapi.Kernel kernel) {
    return kernel;
  }

  static final ResourcePool<com.aparapi.device.Device> devicePool = new ResourcePool<com.aparapi.device.Device>(16) {
    @Override
    public com.aparapi.device.Device create() {
      return KernelManager.instance().bestDevice();
    }
  };}