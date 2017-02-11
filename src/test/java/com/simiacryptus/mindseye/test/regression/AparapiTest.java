package com.simiacryptus.mindseye.test.regression;
/*
Copyright (c) 2010-2011, Advanced Micro Devices, Inc.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:
Redistributions of source code must retain the above copyright notice, this list of conditions and the following
disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
If you use the software (in whole or in part), you shall adhere to all applicable U.S., European, and other export
laws, including but not limited to the U.S. Export Administration Regulations ("EAR"), (15 C.F.R. Sections 730 through
774), and E.U. Council Regulation (EC) No 1334/2000 of 22 June 2000.  Further, pursuant to Section 740.6 of the EAR,
you hereby certify that, except pursuant to a license granted by the United States Department of Commerce Bureau of
Industry and Security or as otherwise permitted pursuant to a License Exception under the U.S. Export Administration
Regulations ("EAR"), you will not (1) export, re-export or release to a national of a country in Country Groups D:1,
E:1 or E:2 any restricted technology, software, or source code you receive hereunder, or (2) export to Country Groups
D:1, E:1 or E:2 the direct product of such technology or software, if such foreign produced direct product is subject
to national security controls as identified on the Commerce Control List (currently found in Supplement 1 to Part 774
of EAR).  For the most current Country Group listings, or for additional information about the EAR or your obligations
under those regulations, please refer to the U.S. Bureau of Industry and Security's website at http://www.bis.doc.gov/.
*/

import java.util.List;

import com.aparapi.Kernel.EXECUTION_MODE;
import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.opencl.OpenCLPlatform;

public class AparapiTest {

  @com.aparapi.opencl.OpenCL.Resource("com/amd/aparapi/sample/convolution/convolution.cl")
  interface Convolution extends com.aparapi.opencl.OpenCL<Convolution> {
    Convolution applyConvolution(//
        com.aparapi.Range range, //
        @GlobalReadOnly("_convMatrix3x3") float[] _convMatrix3x3, //// only read
                                                                  //// from
                                                                  //// kernel
        @GlobalReadOnly("_imagIn") byte[] _imageIn, // only read from kernel
                                                    // (actually char[])
        @GlobalWriteOnly("_imagOut") byte[] _imageOut, // only written to (never
                                                       // read) from kernel
                                                       // (actually char[])
        @Arg("_width") int _width, //
        @Arg("_height") int _height);
  }

  public static class TestKernel extends com.aparapi.Kernel {

    public final int[] input = new int[10240];
    public final int[] results = new int[10240];

    @Override
    public void run() {
      final int i = getGlobalId();
      if (i > 1) {
        this.results[i] += (1 + this.results[i - 1] + this.results[i + 1]) * this.input[i];
      }
    }

  }

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
    } else {
      System.out.println("OpenCLDevice.best() returned { ");
      System.out.println("   Type                  : " + bestDevice.getType());
      System.out.println("   GlobalMemSize         : " + ((OpenCLDevice) bestDevice).getGlobalMemSize());
      System.out.println("   LocalMemSize          : " + ((OpenCLDevice) bestDevice).getLocalMemSize());
      System.out.println("   MaxComputeUnits       : " + ((OpenCLDevice) bestDevice).getMaxComputeUnits());
      System.out.println("   MaxWorkGroupSizes     : " + ((OpenCLDevice) bestDevice).getMaxWorkGroupSize());
      System.out.println("   MaxWorkItemDimensions : " + ((OpenCLDevice) bestDevice).getMaxWorkItemDimensions());
      System.out.println("}");
    }

    final Device firstCPU = Device.firstCPU();
    if (firstCPU == null) {
      System.out.println("OpenCLDevice.firstCPU() returned null!");
    } else {
      System.out.println("OpenCLDevice.firstCPU() returned { ");
      System.out.println("   Type                  : " + firstCPU.getType());
      System.out.println("   GlobalMemSize         : " + ((OpenCLDevice) firstCPU).getGlobalMemSize());
      System.out.println("   LocalMemSize          : " + ((OpenCLDevice) firstCPU).getLocalMemSize());
      System.out.println("   MaxComputeUnits       : " + ((OpenCLDevice) firstCPU).getMaxComputeUnits());
      System.out.println("   MaxWorkGroupSizes     : " + ((OpenCLDevice) firstCPU).getMaxWorkGroupSize());
      System.out.println("   MaxWorkItemDimensions : " + ((OpenCLDevice) firstCPU).getMaxWorkItemDimensions());
      System.out.println("}");
    }

    final Device firstGPU = Device.firstGPU();
    if (firstGPU == null) {
      System.out.println("OpenCLDevice.firstGPU() returned null!");
    } else {
      System.out.println("OpenCLDevice.firstGPU() returned { ");
      System.out.println("   Type                  : " + firstGPU.getType());
      System.out.println("   GlobalMemSize         : " + ((OpenCLDevice) firstGPU).getGlobalMemSize());
      System.out.println("   LocalMemSize          : " + ((OpenCLDevice) firstGPU).getLocalMemSize());
      System.out.println("   MaxComputeUnits       : " + ((OpenCLDevice) firstGPU).getMaxComputeUnits());
      System.out.println("   MaxWorkGroupSizes     : " + ((OpenCLDevice) firstGPU).getMaxWorkGroupSize());
      System.out.println("   MaxWorkItemDimensions : " + ((OpenCLDevice) firstGPU).getMaxWorkItemDimensions());
      System.out.println("}");
    }

    final Device bestGPU = Device.bestGPU();
    if (bestGPU == null) {
      System.out.println("OpenCLDevice.bestGPU() returned null!");
    } else {
      System.out.println("OpenCLDevice.bestGPU() returned { ");
      System.out.println("   Type                  : " + bestGPU.getType());
      System.out.println("   GlobalMemSize         : " + ((OpenCLDevice) bestGPU).getGlobalMemSize());
      System.out.println("   LocalMemSize          : " + ((OpenCLDevice) bestGPU).getLocalMemSize());
      System.out.println("   MaxComputeUnits       : " + ((OpenCLDevice) bestGPU).getMaxComputeUnits());
      System.out.println("   MaxWorkGroupSizes     : " + ((OpenCLDevice) bestGPU).getMaxWorkGroupSize());
      System.out.println("   MaxWorkItemDimensions : " + ((OpenCLDevice) bestGPU).getMaxWorkItemDimensions());
      System.out.println("}");
    }

    final Device firstACC = Device.bestACC();
    if (firstACC == null) {
      System.out.println("OpenCLDevice.firstACC() returned null!");
    } else {
      System.out.println("OpenCLDevice.firstACC() returned { ");
      System.out.println("   Type                  : " + firstACC.getType());
      System.out.println("   GlobalMemSize         : " + ((OpenCLDevice) firstACC).getGlobalMemSize());
      System.out.println("   LocalMemSize          : " + ((OpenCLDevice) firstACC).getLocalMemSize());
      System.out.println("   MaxComputeUnits       : " + ((OpenCLDevice) firstACC).getMaxComputeUnits());
      System.out.println("   MaxWorkGroupSizes     : " + ((OpenCLDevice) firstACC).getMaxWorkGroupSize());
      System.out.println("   MaxWorkItemDimensions : " + ((OpenCLDevice) firstACC).getMaxWorkItemDimensions());
      System.out.println("}");
    }

    final Device bestACC = Device.bestACC();
    if (bestACC == null) {
      System.out.println("OpenCLDevice.bestACC() returned null!");
    } else {
      System.out.println("OpenCLDevice.bestACC() returned { ");
      System.out.println("   Type                  : " + bestACC.getType());
      System.out.println("   GlobalMemSize         : " + ((OpenCLDevice) bestACC).getGlobalMemSize());
      System.out.println("   LocalMemSize          : " + ((OpenCLDevice) bestACC).getLocalMemSize());
      System.out.println("   MaxComputeUnits       : " + ((OpenCLDevice) bestACC).getMaxComputeUnits());
      System.out.println("   MaxWorkGroupSizes     : " + ((OpenCLDevice) bestACC).getMaxWorkGroupSize());
      System.out.println("   MaxWorkItemDimensions : " + ((OpenCLDevice) bestACC).getMaxWorkItemDimensions());
      System.out.println("}");
    }

  }

  public AparapiTest() {
    super();
  }

  @org.junit.Test
  public void test() {

    final com.aparapi.device.OpenCLDevice openclDevice = (com.aparapi.device.OpenCLDevice) com.aparapi.device.Device.best();
    // final Convolution convolution = openclDevice.bind(Convolution.class);
    final TestKernel testKernel = new TestKernel();
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

}
