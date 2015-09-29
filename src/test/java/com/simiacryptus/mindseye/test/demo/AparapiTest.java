package com.simiacryptus.mindseye.test.demo;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

import org.junit.Test;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.amd.aparapi.Range;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.basic.EntropyLossLayer;
import com.simiacryptus.mindseye.net.basic.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.basic.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.basic.SqLossLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.net.dev.MinMaxFilterLayer;
import com.simiacryptus.mindseye.test.Tester;
import com.simiacryptus.mindseye.test.dev.MNIST;
import com.simiacryptus.mindseye.test.dev.SimpleMNIST;
import com.simiacryptus.mindseye.training.DevelopmentTrainer;
import com.simiacryptus.mindseye.training.DynamicRateTrainer;
import com.simiacryptus.mindseye.training.GradientDescentTrainer;
import com.simiacryptus.mindseye.training.NetInitializer;
import com.simiacryptus.mindseye.util.LabeledObject;
import com.simiacryptus.mindseye.util.Util;

public class AparapiTest  {

  @com.amd.aparapi.opencl.OpenCL.Resource("com/amd/aparapi/sample/convolution/convolution.cl") interface Convolution extends com.amd.aparapi.opencl.OpenCL<Convolution>{
     Convolution applyConvolution(//
           com.amd.aparapi.Range range, //
           @GlobalReadOnly("_convMatrix3x3") float[] _convMatrix3x3,//// only read from kernel 
           @GlobalReadOnly("_imagIn") byte[] _imageIn,// only read from kernel (actually char[])
           @GlobalWriteOnly("_imagOut") byte[] _imageOut, // only written to (never read) from kernel (actually char[])
           @Arg("_width") int _width,// 
           @Arg("_height") int _height);
  }

  public static class TestKernel extends com.amd.aparapi.Kernel {

    public final int[] input = new int[10240];
    public final int[] results = new int[10240];
    
    @Override
    public void run() {
      int i = getGlobalId();
      if(i>1)
      {
        results[i] += (1 + results[i-1] + results[i+1])*input[i];
      }
    }
    
  }
  
  public AparapiTest() {
    super();
  }
  
  @org.junit.Test
  public void test(){

    final com.amd.aparapi.device.OpenCLDevice openclDevice = (com.amd.aparapi.device.OpenCLDevice) com.amd.aparapi.device.Device.best();
    //final Convolution convolution = openclDevice.bind(Convolution.class);
    TestKernel testKernel = new TestKernel();
    testKernel.setExecutionMode(EXECUTION_MODE.GPU);
    testKernel.setExplicit(true);
    Range range = openclDevice.createRange3D(100, 100,8);
    for(int j=0;j<20480;j++)
    {
      testKernel.put(testKernel.input);
      testKernel.execute(range);
      testKernel.get(testKernel.results);
      System.out.println("OK:"+j);
    }
    testKernel.dispose();
  }


}
