/*
 * Copyright (c) 2017 by Andrew Charneski.
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

import java.util.List;
import java.util.Random;

/**
 * The type Aparapi test.
 */
public class AparapiTest {
  /**
   * The constant randomize.
   */
  public static final Random random = new Random();
  /**
   * The Log.
   */
  static final Logger log = LoggerFactory.getLogger(AparapiTest.class);
  private static final Logger logger = LoggerFactory.getLogger(AparapiTest.class);
  
  /**
   * Instantiates a new Aparapi test.
   */
  public AparapiTest() {
    super();
  }
  
  /**
   * Main.
   *
   * @param _args the args
   */
  public static void main(final String[] _args) {
    logger.info("com.amd.aparapi.sample.info.Main");
    final List<OpenCLPlatform> platforms = new OpenCLPlatform().getOpenCLPlatforms();
    logger.info("Machine contains " + platforms.size() + " OpenCL platforms");
    int platformc = 0;
    for (final OpenCLPlatform platform : platforms) {
      logger.info("Platform " + platformc + "{");
      logger.info("   Name    : \"" + platform.getName() + "\"");
      logger.info("   Vendor  : \"" + platform.getVendor() + "\"");
      logger.info("   Version : \"" + platform.getVersion() + "\"");
      final List<OpenCLDevice> devices = platform.getOpenCLDevices();
      logger.info("   Platform contains " + devices.size() + " OpenCL devices");
      int devicec = 0;
      for (final OpenCLDevice device : devices) {
        logger.info("   Device " + devicec + "{");
        logger.info("       Type                  : " + device.getType());
        logger.info("       GlobalMemSize         : " + device.getGlobalMemSize());
        logger.info("       LocalMemSize          : " + device.getLocalMemSize());
        logger.info("       MaxComputeUnits       : " + device.getMaxComputeUnits());
        logger.info("       MaxWorkGroupSizes     : " + device.getMaxWorkGroupSize());
        logger.info("       MaxWorkItemDimensions : " + device.getMaxWorkItemDimensions());
        logger.info("   }");
        devicec++;
      }
      logger.info("}");
      platformc++;
    }

    final Device bestDevice = Device.best();
    if (bestDevice == null) {
      logger.info("OpenCLDevice.best() returned null!");
    }
    else {
      logger.info("OpenCLDevice.best() returned { ");
      logger.info("   Type                  : " + bestDevice.getType());
      logger.info("   GlobalMemSize         : " + ((OpenCLDevice) bestDevice).getGlobalMemSize());
      logger.info("   LocalMemSize          : " + ((OpenCLDevice) bestDevice).getLocalMemSize());
      logger.info("   MaxComputeUnits       : " + ((OpenCLDevice) bestDevice).getMaxComputeUnits());
      logger.info("   MaxWorkGroupSizes     : " + bestDevice.getMaxWorkGroupSize());
      logger.info("   MaxWorkItemDimensions : " + bestDevice.getMaxWorkItemDimensions());
      logger.info("}");
    }

    final Device firstCPU = Device.firstCPU();
    if (firstCPU == null) {
      logger.info("OpenCLDevice.firstCPU() returned null!");
    }
    else {
      logger.info("OpenCLDevice.firstCPU() returned { ");
      logger.info("   Type                  : " + firstCPU.getType());
      logger.info("   GlobalMemSize         : " + ((OpenCLDevice) firstCPU).getGlobalMemSize());
      logger.info("   LocalMemSize          : " + ((OpenCLDevice) firstCPU).getLocalMemSize());
      logger.info("   MaxComputeUnits       : " + ((OpenCLDevice) firstCPU).getMaxComputeUnits());
      logger.info("   MaxWorkGroupSizes     : " + firstCPU.getMaxWorkGroupSize());
      logger.info("   MaxWorkItemDimensions : " + firstCPU.getMaxWorkItemDimensions());
      logger.info("}");
    }

    final Device firstGPU = Device.firstGPU();
    if (firstGPU == null) {
      logger.info("OpenCLDevice.firstGPU() returned null!");
    }
    else {
      logger.info("OpenCLDevice.firstGPU() returned { ");
      logger.info("   Type                  : " + firstGPU.getType());
      logger.info("   GlobalMemSize         : " + ((OpenCLDevice) firstGPU).getGlobalMemSize());
      logger.info("   LocalMemSize          : " + ((OpenCLDevice) firstGPU).getLocalMemSize());
      logger.info("   MaxComputeUnits       : " + ((OpenCLDevice) firstGPU).getMaxComputeUnits());
      logger.info("   MaxWorkGroupSizes     : " + firstGPU.getMaxWorkGroupSize());
      logger.info("   MaxWorkItemDimensions : " + firstGPU.getMaxWorkItemDimensions());
      logger.info("}");
    }

    final Device bestGPU = Device.bestGPU();
    if (bestGPU == null) {
      logger.info("OpenCLDevice.bestGPU() returned null!");
    }
    else {
      logger.info("OpenCLDevice.bestGPU() returned { ");
      logger.info("   Type                  : " + bestGPU.getType());
      logger.info("   GlobalMemSize         : " + ((OpenCLDevice) bestGPU).getGlobalMemSize());
      logger.info("   LocalMemSize          : " + ((OpenCLDevice) bestGPU).getLocalMemSize());
      logger.info("   MaxComputeUnits       : " + ((OpenCLDevice) bestGPU).getMaxComputeUnits());
      logger.info("   MaxWorkGroupSizes     : " + bestGPU.getMaxWorkGroupSize());
      logger.info("   MaxWorkItemDimensions : " + bestGPU.getMaxWorkItemDimensions());
      logger.info("}");
    }
  
    final Device firstACC = Device.bestACC();
    if (firstACC == null) {
      logger.info("OpenCLDevice.firstACC() returned null!");
    }
    else {
      logger.info("OpenCLDevice.firstACC() returned { ");
      logger.info("   Type                  : " + firstACC.getType());
      logger.info("   GlobalMemSize         : " + ((OpenCLDevice) firstACC).getGlobalMemSize());
      logger.info("   LocalMemSize          : " + ((OpenCLDevice) firstACC).getLocalMemSize());
      logger.info("   MaxComputeUnits       : " + ((OpenCLDevice) firstACC).getMaxComputeUnits());
      logger.info("   MaxWorkGroupSizes     : " + firstACC.getMaxWorkGroupSize());
      logger.info("   MaxWorkItemDimensions : " + firstACC.getMaxWorkItemDimensions());
      logger.info("}");
    }
  
    final Device bestACC = Device.bestACC();
    if (bestACC == null) {
      logger.info("OpenCLDevice.bestACC() returned null!");
    }
    else {
      logger.info("OpenCLDevice.bestACC() returned { ");
      logger.info("   Type                  : " + bestACC.getType());
      logger.info("   GlobalMemSize         : " + ((OpenCLDevice) bestACC).getGlobalMemSize());
      logger.info("   LocalMemSize          : " + ((OpenCLDevice) bestACC).getLocalMemSize());
      logger.info("   MaxComputeUnits       : " + ((OpenCLDevice) bestACC).getMaxComputeUnits());
      logger.info("   MaxWorkGroupSizes     : " + bestACC.getMaxWorkGroupSize());
      logger.info("   MaxWorkItemDimensions : " + bestACC.getMaxWorkItemDimensions());
      logger.info("}");
    }
  
  }
  
  /**
   * Test 1.
   */
  @Test
  @Ignore
  public void test1() {
  
    final OpenCLDevice openclDevice = (OpenCLDevice) Device.best();
    // final Convolution convolution = openclDevice.bind(Convolution.class);
    final AparapiTest.TestKernel testKernel = new AparapiTest.TestKernel();
    testKernel.setExecutionMode(EXECUTION_MODE.GPU);
    testKernel.setExplicit(true);
    final Range range = openclDevice.createRange3D(100, 100, 8);
    for (int j = 0; j < 2048; j++) {
      testKernel.put(testKernel.input);
      testKernel.execute(range);
      testKernel.get(testKernel.results);
      logger.info("OK:" + j);
    }
    testKernel.dispose();
  }
  
  /**
   * Test 2.
   *
   * @throws Exception the exception
   */
  @Test
  public void test2() throws Exception {
    final float inA[] = new float[1024];
    final float inB[] = new float[1024];
    assert inA.length == inB.length;
    final float[] result = new float[inA.length];
  
    final Kernel kernel = new Kernel() {
      @Override
      public void run() {
        final int i = getGlobalId();
        result[i] = inA[i] + inB[i];
      }
    };
  
    final Range range = Range.create(result.length);
    kernel.execute(range);
  }
  
  /**
   * The interface Convolution.
   */
  @Resource("com/amd/aparapi/sample/convolution/convolution.cl")
  interface Convolution extends com.aparapi.opencl.OpenCL<AparapiTest.Convolution> {
    /**
     * Apply convolution aparapi test . convolution.
     *
     * @param range          the range
     * @param _convMatrix3x3 the conv matrix 3 x 3
     * @param _imageIn       the image in
     * @param _imageOut      the image out
     * @param _width         the width
     * @param _height        the height
     * @return the aparapi test . convolution
     */
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
