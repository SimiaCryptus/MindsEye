package com.simiacryptus.mindseye.opencl;

import com.aparapi.Kernel;
import com.aparapi.Kernel.EXECUTION_MODE;
import com.aparapi.device.Device.TYPE;

public final class OpenCL{

  public static EXECUTION_MODE getExecutionMode() {
    return EXECUTION_MODE.CPU;
  }

  public static Kernel init(final com.aparapi.Kernel kernel) {
    kernel.setExecutionMode(EXECUTION_MODE.CPU);
    kernel.addExecutionModes(EXECUTION_MODE.CPU, EXECUTION_MODE.GPU, EXECUTION_MODE.SEQ);
    return kernel;
  }

  static final ResourcePool<com.aparapi.device.Device> range = new ResourcePool<com.aparapi.device.Device>(16) {
    @Override
    public com.aparapi.device.Device create() {
      //if(1==1) return com.amd.aparapi.device.Device.best();
      com.aparapi.device.Device openclDevice;
      if (getExecutionMode() == EXECUTION_MODE.CPU) {
        openclDevice = com.aparapi.device.Device.firstCPU();
      } else if (getExecutionMode() == EXECUTION_MODE.ACC) {
        openclDevice = com.aparapi.device.Device.bestACC();
      } else if (getExecutionMode() == EXECUTION_MODE.GPU) {
        openclDevice = com.aparapi.device.Device.bestGPU();
      } else {
        openclDevice = com.aparapi.device.Device.first(TYPE.SEQ);
        if (null == openclDevice) {
          openclDevice = com.aparapi.device.Device.firstCPU();
          openclDevice.setType(TYPE.SEQ);
        }
      }
      return openclDevice;
    }
  };}