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
    System.out.println("com.amd.aparapi.sample.info.Main");
    final List<OpenCLPlatform> platforms = new OpenCLPlatform().getOpenCLPlatforms();
    System.out.println("Machine contains " + platforms.size() + " OpenCL platforms");
    int platformc = 0;
    for (final OpenCLPlatform platform : platforms) {
      System.out.println("Platform " + platformc + "{");
      System.out.println("   Name    : \"" + platform.getName() + "\"");
      System.out.println("   Vendor  : \"" + platform.getVendor() + "\"");
      System.out.println("   Version : \"" + platform.getVersion() + "\"");
      final List<OpenCLDevice> devices = platform.getOpenCLDevices();
      System.out.println("   Platform contains " + devices.size() + " OpenCL devices");
      int devicec = 0;
      for (final OpenCLDevice device : devices) {
        System.out.println("   Device " + devicec + "{");
        System.out.println("       Type                  : " + device.getType());
        System.out.println("       GlobalMemSize         : " + device.getGlobalMemSize());
        System.out.println("       LocalMemSize          : " + device.getLocalMemSize());
        System.out.println("       MaxComputeUnits       : " + device.getMaxComputeUnits());
        System.out.println("       MaxWorkGroupSizes     : " + device.getMaxWorkGroupSize());
        System.out.println("       MaxWorkItemDimensions : " + device.getMaxWorkItemDimensions());
        System.out.println("   }");
        devicec++;
      }
      System.out.println("}");
      platformc++;
    }
    
    final Device bestDevice = Device.best();
    if (bestDevice == null) {
      System.out.println("OpenCLDevice.best() returned null!");
    }
    else {
      System.out.println("OpenCLDevice.best() returned { ");
      System.out.println("   Type                  : " + bestDevice.getType());
      System.out.println("   GlobalMemSize         : " + ((OpenCLDevice) bestDevice).getGlobalMemSize());
      System.out.println("   LocalMemSize          : " + ((OpenCLDevice) bestDevice).getLocalMemSize());
      System.out.println("   MaxComputeUnits       : " + ((OpenCLDevice) bestDevice).getMaxComputeUnits());
      System.out.println("   MaxWorkGroupSizes     : " + bestDevice.getMaxWorkGroupSize());
      System.out.println("   MaxWorkItemDimensions : " + bestDevice.getMaxWorkItemDimensions());
      System.out.println("}");
    }
    
    final Device firstCPU = Device.firstCPU();
    if (firstCPU == null) {
      System.out.println("OpenCLDevice.firstCPU() returned null!");
    }
    else {
      System.out.println("OpenCLDevice.firstCPU() returned { ");
      System.out.println("   Type                  : " + firstCPU.getType());
      System.out.println("   GlobalMemSize         : " + ((OpenCLDevice) firstCPU).getGlobalMemSize());
      System.out.println("   LocalMemSize          : " + ((OpenCLDevice) firstCPU).getLocalMemSize());
      System.out.println("   MaxComputeUnits       : " + ((OpenCLDevice) firstCPU).getMaxComputeUnits());
      System.out.println("   MaxWorkGroupSizes     : " + firstCPU.getMaxWorkGroupSize());
      System.out.println("   MaxWorkItemDimensions : " + firstCPU.getMaxWorkItemDimensions());
      System.out.println("}");
    }
    
    final Device firstGPU = Device.firstGPU();
    if (firstGPU == null) {
      System.out.println("OpenCLDevice.firstGPU() returned null!");
    }
    else {
      System.out.println("OpenCLDevice.firstGPU() returned { ");
      System.out.println("   Type                  : " + firstGPU.getType());
      System.out.println("   GlobalMemSize         : " + ((OpenCLDevice) firstGPU).getGlobalMemSize());
      System.out.println("   LocalMemSize          : " + ((OpenCLDevice) firstGPU).getLocalMemSize());
      System.out.println("   MaxComputeUnits       : " + ((OpenCLDevice) firstGPU).getMaxComputeUnits());
      System.out.println("   MaxWorkGroupSizes     : " + firstGPU.getMaxWorkGroupSize());
      System.out.println("   MaxWorkItemDimensions : " + firstGPU.getMaxWorkItemDimensions());
      System.out.println("}");
    }
    
    final Device bestGPU = Device.bestGPU();
    if (bestGPU == null) {
      System.out.println("OpenCLDevice.bestGPU() returned null!");
    }
    else {
      System.out.println("OpenCLDevice.bestGPU() returned { ");
      System.out.println("   Type                  : " + bestGPU.getType());
      System.out.println("   GlobalMemSize         : " + ((OpenCLDevice) bestGPU).getGlobalMemSize());
      System.out.println("   LocalMemSize          : " + ((OpenCLDevice) bestGPU).getLocalMemSize());
      System.out.println("   MaxComputeUnits       : " + ((OpenCLDevice) bestGPU).getMaxComputeUnits());
      System.out.println("   MaxWorkGroupSizes     : " + bestGPU.getMaxWorkGroupSize());
      System.out.println("   MaxWorkItemDimensions : " + bestGPU.getMaxWorkItemDimensions());
      System.out.println("}");
    }
    
    final Device firstACC = Device.bestACC();
    if (firstACC == null) {
      System.out.println("OpenCLDevice.firstACC() returned null!");
    }
    else {
      System.out.println("OpenCLDevice.firstACC() returned { ");
      System.out.println("   Type                  : " + firstACC.getType());
      System.out.println("   GlobalMemSize         : " + ((OpenCLDevice) firstACC).getGlobalMemSize());
      System.out.println("   LocalMemSize          : " + ((OpenCLDevice) firstACC).getLocalMemSize());
      System.out.println("   MaxComputeUnits       : " + ((OpenCLDevice) firstACC).getMaxComputeUnits());
      System.out.println("   MaxWorkGroupSizes     : " + firstACC.getMaxWorkGroupSize());
      System.out.println("   MaxWorkItemDimensions : " + firstACC.getMaxWorkItemDimensions());
      System.out.println("}");
    }
    
    final Device bestACC = Device.bestACC();
    if (bestACC == null) {
      System.out.println("OpenCLDevice.bestACC() returned null!");
    }
    else {
      System.out.println("OpenCLDevice.bestACC() returned { ");
      System.out.println("   Type                  : " + bestACC.getType());
      System.out.println("   GlobalMemSize         : " + ((OpenCLDevice) bestACC).getGlobalMemSize());
      System.out.println("   LocalMemSize          : " + ((OpenCLDevice) bestACC).getLocalMemSize());
      System.out.println("   MaxComputeUnits       : " + ((OpenCLDevice) bestACC).getMaxComputeUnits());
      System.out.println("   MaxWorkGroupSizes     : " + bestACC.getMaxWorkGroupSize());
      System.out.println("   MaxWorkItemDimensions : " + bestACC.getMaxWorkItemDimensions());
      System.out.println("}");
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
      System.out.println("OK:" + j);
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
    float inA[] = new float[1024];
    float inB[] = new float[1024];
    assert (inA.length == inB.length);
    float[] result = new float[inA.length];
    
    Kernel kernel = new Kernel() {
      @Override
      public void run() {
        int i = getGlobalId();
        result[i] = inA[i] + inB[i];
      }
    };
    
    Range range = Range.create(result.length);
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
        this.results[i] += (1 + this.results[i - 1] + this.results[i + 1]) * this.input[i];
      }
    }
    
  }
  
  
}
