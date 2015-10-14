package com.simiacryptus.mindseye.opencl;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.amd.aparapi.device.Device.TYPE;

public final class OpenCL{

  public static EXECUTION_MODE getExecutionMode() {
    return EXECUTION_MODE.CPU;
  }

  public static Kernel init(final com.amd.aparapi.Kernel kernel) {
    kernel.setExecutionMode(EXECUTION_MODE.CPU);
    kernel.addExecutionModes(EXECUTION_MODE.CPU, EXECUTION_MODE.GPU, EXECUTION_MODE.SEQ);
    return kernel;
  }

  static final ResourcePool<com.amd.aparapi.device.Device> range = new ResourcePool<com.amd.aparapi.device.Device>(16) {
    @Override
    public com.amd.aparapi.device.Device create() {
      //if(1==1) return com.amd.aparapi.device.Device.best();
      com.amd.aparapi.device.Device openclDevice;
      if (getExecutionMode() == EXECUTION_MODE.CPU) {
        openclDevice = com.amd.aparapi.device.Device.firstCPU();
      } else if (getExecutionMode() == EXECUTION_MODE.ACC) {
        openclDevice = com.amd.aparapi.device.Device.firstACC();
      } else if (getExecutionMode() == EXECUTION_MODE.GPU) {
        openclDevice = com.amd.aparapi.device.Device.bestGPU();
      } else {
        openclDevice = com.amd.aparapi.device.Device.first(TYPE.SEQ);
        if (null == openclDevice) {
          openclDevice = com.amd.aparapi.device.Device.firstCPU();
          openclDevice.setType(TYPE.SEQ);
        }
      }
      return openclDevice;
    }
  };}