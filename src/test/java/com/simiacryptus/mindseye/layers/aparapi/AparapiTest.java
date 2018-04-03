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

import com.aparapi.Kernel;
import com.aparapi.Kernel.EXECUTION_MODE;
import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.opencl.OpenCLPlatform;
import com.aparapi.opencl.OpenCL;
import com.aparapi.opencl.OpenCL.Resource;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.List;
import java.util.Random;

/**
 * The type Aparapi apply.
 */
public class AparapiTest {
  /**
   * The constant randomize.
   */
  public static final Random random = new Random();
  /**
   * The Log.
   */
  private static final Logger log = LoggerFactory.getLogger(AparapiTest.class);
  
  /**
   * Instantiates a new Aparapi apply.
   */
  public AparapiTest() {
    super();
  }
  
  /**
   * Main.
   *
   * @param _args the args
   */
  public static void main(final CharSequence[] _args) {
    log.info("com.amd.aparapi.sample.info.Main");
    final List<OpenCLPlatform> platforms = new OpenCLPlatform().getOpenCLPlatforms();
    log.info("Machine contains " + platforms.size() + " OpenCL platforms");
    int platformc = 0;
    for (@Nonnull final OpenCLPlatform platform : platforms) {
      log.info("Platform " + platformc + "{");
      log.info("   Name    : \"" + platform.getName() + "\"");
      log.info("   Vendor  : \"" + platform.getVendor() + "\"");
      log.info("   Version : \"" + platform.getVersion() + "\"");
      final List<OpenCLDevice> devices = platform.getOpenCLDevices();
      log.info("   Platform contains " + devices.size() + " OpenCL devices");
      int devicec = 0;
      for (@Nonnull final OpenCLDevice device : devices) {
        log.info("   Device " + devicec + "{");
        log.info("       Type                  : " + device.getType());
        log.info("       GlobalMemSize         : " + device.getGlobalMemSize());
        log.info("       LocalMemSize          : " + device.getLocalMemSize());
        log.info("       MaxComputeUnits       : " + device.getMaxComputeUnits());
        log.info("       MaxWorkGroupSizes     : " + device.getMaxWorkGroupSize());
        log.info("       MaxWorkItemDimensions : " + device.getMaxWorkItemDimensions());
        log.info("   }");
        devicec++;
      }
      log.info("}");
      platformc++;
    }
  
    final Device bestDevice = Device.best();
    if (bestDevice == null) {
      log.info("OpenCLDevice.best() returned null!");
    }
    else {
      log.info("OpenCLDevice.best() returned { ");
      log.info("   Type                  : " + bestDevice.getType());
      log.info("   GlobalMemSize         : " + ((OpenCLDevice) bestDevice).getGlobalMemSize());
      log.info("   LocalMemSize          : " + ((OpenCLDevice) bestDevice).getLocalMemSize());
      log.info("   MaxComputeUnits       : " + ((OpenCLDevice) bestDevice).getMaxComputeUnits());
      log.info("   MaxWorkGroupSizes     : " + bestDevice.getMaxWorkGroupSize());
      log.info("   MaxWorkItemDimensions : " + bestDevice.getMaxWorkItemDimensions());
      log.info("}");
    }
  
    final Device firstCPU = Device.firstCPU();
    if (firstCPU == null) {
      log.info("OpenCLDevice.firstCPU() returned null!");
    }
    else {
      log.info("OpenCLDevice.firstCPU() returned { ");
      log.info("   Type                  : " + firstCPU.getType());
      log.info("   GlobalMemSize         : " + ((OpenCLDevice) firstCPU).getGlobalMemSize());
      log.info("   LocalMemSize          : " + ((OpenCLDevice) firstCPU).getLocalMemSize());
      log.info("   MaxComputeUnits       : " + ((OpenCLDevice) firstCPU).getMaxComputeUnits());
      log.info("   MaxWorkGroupSizes     : " + firstCPU.getMaxWorkGroupSize());
      log.info("   MaxWorkItemDimensions : " + firstCPU.getMaxWorkItemDimensions());
      log.info("}");
    }
  
    final Device firstGPU = Device.firstGPU();
    if (firstGPU == null) {
      log.info("OpenCLDevice.firstGPU() returned null!");
    }
    else {
      log.info("OpenCLDevice.firstGPU() returned { ");
      log.info("   Type                  : " + firstGPU.getType());
      log.info("   GlobalMemSize         : " + ((OpenCLDevice) firstGPU).getGlobalMemSize());
      log.info("   LocalMemSize          : " + ((OpenCLDevice) firstGPU).getLocalMemSize());
      log.info("   MaxComputeUnits       : " + ((OpenCLDevice) firstGPU).getMaxComputeUnits());
      log.info("   MaxWorkGroupSizes     : " + firstGPU.getMaxWorkGroupSize());
      log.info("   MaxWorkItemDimensions : " + firstGPU.getMaxWorkItemDimensions());
      log.info("}");
    }
  
    final Device bestGPU = Device.bestGPU();
    if (bestGPU == null) {
      log.info("OpenCLDevice.bestGPU() returned null!");
    }
    else {
      log.info("OpenCLDevice.bestGPU() returned { ");
      log.info("   Type                  : " + bestGPU.getType());
      log.info("   GlobalMemSize         : " + ((OpenCLDevice) bestGPU).getGlobalMemSize());
      log.info("   LocalMemSize          : " + ((OpenCLDevice) bestGPU).getLocalMemSize());
      log.info("   MaxComputeUnits       : " + ((OpenCLDevice) bestGPU).getMaxComputeUnits());
      log.info("   MaxWorkGroupSizes     : " + bestGPU.getMaxWorkGroupSize());
      log.info("   MaxWorkItemDimensions : " + bestGPU.getMaxWorkItemDimensions());
      log.info("}");
    }
  
    final Device firstACC = Device.bestACC();
    if (firstACC == null) {
      log.info("OpenCLDevice.firstACC() returned null!");
    }
    else {
      log.info("OpenCLDevice.firstACC() returned { ");
      log.info("   Type                  : " + firstACC.getType());
      log.info("   GlobalMemSize         : " + ((OpenCLDevice) firstACC).getGlobalMemSize());
      log.info("   LocalMemSize          : " + ((OpenCLDevice) firstACC).getLocalMemSize());
      log.info("   MaxComputeUnits       : " + ((OpenCLDevice) firstACC).getMaxComputeUnits());
      log.info("   MaxWorkGroupSizes     : " + firstACC.getMaxWorkGroupSize());
      log.info("   MaxWorkItemDimensions : " + firstACC.getMaxWorkItemDimensions());
      log.info("}");
    }
  
    final Device bestACC = Device.bestACC();
    if (bestACC == null) {
      log.info("OpenCLDevice.bestACC() returned null!");
    }
    else {
      log.info("OpenCLDevice.bestACC() returned { ");
      log.info("   Type                  : " + bestACC.getType());
      log.info("   GlobalMemSize         : " + ((OpenCLDevice) bestACC).getGlobalMemSize());
      log.info("   LocalMemSize          : " + ((OpenCLDevice) bestACC).getLocalMemSize());
      log.info("   MaxComputeUnits       : " + ((OpenCLDevice) bestACC).getMaxComputeUnits());
      log.info("   MaxWorkGroupSizes     : " + bestACC.getMaxWorkGroupSize());
      log.info("   MaxWorkItemDimensions : " + bestACC.getMaxWorkItemDimensions());
      log.info("}");
    }
  
  }
  
  /**
   * Test 1.
   */
  @Test
  @Ignore
  public void test1() {
  
    @Nonnull final OpenCLDevice openclDevice = (OpenCLDevice) Device.best();
    // final Convolution convolution = openclDevice.bind(Convolution.class);
    @Nonnull final AparapiTest.TestKernel testKernel = new AparapiTest.TestKernel();
    testKernel.setExecutionMode(EXECUTION_MODE.GPU);
    testKernel.setExplicit(true);
    final Range range = openclDevice.createRange3D(100, 100, 8);
    for (int j = 0; j < 2048; j++) {
      testKernel.put(testKernel.input);
      testKernel.execute(range);
      testKernel.get(testKernel.results);
      log.info("OK:" + j);
    }
    testKernel.dispose();
  }
  
  /**
   * Test 2.
   *
   * @throws Exception the exception
   */
  @Test
  public void test2() {
    @Nonnull final float inA[] = new float[1024];
    @Nonnull final float inB[] = new float[1024];
    assert inA.length == inB.length;
    @Nonnull final float[] result = new float[inA.length];
  
    @Nonnull final Kernel kernel = new Kernel() {
      @Override
      public void run() {
        final int i = getGlobalId();
        result[i] = inA[i] + inB[i];
      }
    };
  
    @Nonnull final Range range = Range.create(result.length);
    kernel.execute(range);
  }
  
  /**
   * The interface Convolution.
   */
  @Resource("com/amd/aparapi/sample/convolution/convolution.cl")
  interface Convolution extends com.aparapi.opencl.OpenCL<AparapiTest.Convolution> {
    /**
     * Apply convolution aparapi apply . convolution.
     *
     * @param range          the range
     * @param _convMatrix3x3 the conv matrix 3 x 3
     * @param _imageIn       the image in
     * @param _imageOut      the image out
     * @param _width         the width
     * @param _height        the height
     * @return the aparapi apply . convolution
     */
    @Nonnull
    AparapiTest.Convolution applyConvolution(//
      Range range, //
      @OpenCL.GlobalReadOnly("_convMatrix3x3") float[] _convMatrix3x3, //// only read
      //// from
      //// filter
      @OpenCL.GlobalReadOnly("_imagIn") byte[] _imageIn, // only read from filter
      // (actually char[])
      @OpenCL.GlobalWriteOnly("_imagOut") byte[] _imageOut, // only written to (never
      // read) from filter
      // (actually char[])
      @OpenCL.Arg("_width") int _width, //
      @OpenCL.Arg("_height") int _height);
  }
  
  /**
   * The type Test kernel.
   */
  public static class TestKernel extends Kernel {
  
    /**
     * The Input.
     */
    public final int[] input = new int[10240];
    /**
     * The Results.
     */
    public final int[] results = new int[10240];
  
    @Override
    public void run() {
      final int i = getGlobalId();
      if (i > 1) {
        results[i] += (1 + results[i - 1] + results[i + 1]) * input[i];
      }
    }
  
  }
  
  
}
