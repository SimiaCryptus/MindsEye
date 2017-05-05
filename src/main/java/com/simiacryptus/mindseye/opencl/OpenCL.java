package com.simiacryptus.mindseye.opencl;

import com.aparapi.Kernel;
import com.aparapi.device.Device;
import com.aparapi.internal.kernel.KernelManager;
import com.simiacryptus.util.lang.ResourcePool;

public final class OpenCL{

  static final ResourcePool<Device> devicePool = new ResourcePool<com.aparapi.device.Device>(16) {
    @Override
    public com.aparapi.device.Device create() {
      return KernelManager.instance().bestDevice();
    }
  };}