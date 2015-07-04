package com.simiacryptus.mindseye.test.dev;

import java.awt.Desktop;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.PipelineNetwork;
import com.simiacryptus.mindseye.data.LabeledObject;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.SigmoidActivationLayer;
import com.simiacryptus.mindseye.layers.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.learning.NNResult;

public class ImageNetworkDev {
  static final Logger log = LoggerFactory.getLogger(ImageNetworkDev.class);
  
  public static final Random random = new Random();
  
  @Test
  public void test_BasicNN() throws Exception {
    
    
    final int[] inputSize = new int[] { 28, 28 };
    int[] kernelSize = new int[] { 3, 3 };
    final int[] outSize = new int[]{inputSize[0] - kernelSize[0] + 1,inputSize[1] - kernelSize[1] + 1};

    //List<LabeledObject<NDArray>> data = TestMNISTDev.trainingDataStream().limit(10).collect(Collectors.toList());
    List<LabeledObject<NDArray>> data = new ArrayList<>();
    data.add(new LabeledObject<NDArray>(TestMNISTDev.toNDArray(render(inputSize, "Hello")), ""));
    
    
    ConvolutionSynapseLayer convolution = new ConvolutionSynapseLayer(kernelSize, 1);
    convolution.kernel.set(new int[]{0,2,0}, 1);
    convolution.kernel.set(new int[]{1,1,0}, 1);
    convolution.kernel.set(new int[]{2,0,0}, 1);
    convolution.freeze();
    
    PipelineNetwork net = new PipelineNetwork()
        .add(convolution);
    
    
    Stream<BufferedImage[]> buffer = data.stream().map(obj->{
      NNResult output = net.eval(obj.data);
      BiasLayer bias = new BiasLayer(inputSize);
      PipelineNetwork invnet = new PipelineNetwork()
        .add(bias)
        .add(convolution);
      invnet.setVerbose(true).train(new NDArray[][]{{obj.data, obj.data}}, 10000, 0.00001);

      NNResult recovered = bias.eval(obj.data);
      NNResult tested = net.eval(recovered.data);
      
      return new BufferedImage[]{
          TestMNISTDev.toImage(obj.data),
          TestMNISTDev.toImage(new NDArray(outSize, output.data.data)),
          TestMNISTDev.toImage(new NDArray(inputSize, recovered.data.data)),
          TestMNISTDev.toImage(new NDArray(outSize, tested.data.data))
      };
    });
    
    
    final File outDir = new File("reports");
    outDir.mkdirs();
    final StackTraceElement caller = Thread.currentThread().getStackTrace()[2];
    final File report = new File(outDir, caller.getClassName() + "_" + caller.getLineNumber() + ".html");
    final PrintStream out = new PrintStream(new FileOutputStream(report));
    out.println("<html><head></head><body>");
    buffer.map(x -> "<p>" + Stream.of(x).map(o->TestMNISTDev.toInlineImage(o, "")).reduce((a, b)->a+b) + "</p>")
        .forEach(out::println);
    out.println("</body></html>");
    out.close();
    Desktop.getDesktop().browse(report.toURI());
    
  }

  private BufferedImage render(final int[] inputSize, String string) {
    BufferedImage img = new BufferedImage(inputSize[0], inputSize[1], BufferedImage.TYPE_INT_RGB);
    img.createGraphics().drawString(string, 0, 13);
    return img;
  }
  
  @Test
  public void test_BasicNN_AND() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] midSize = new int[] { 3 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        // XOR:
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { -1 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1 }) }
    };
    new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize).addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian()))
        .add(new BiasLayer(midSize))
        .add(new SigmoidActivationLayer())
        
        .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize).addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian()))
        .add(new BiasLayer(outSize))
        .add(new SigmoidActivationLayer())
        .setRate(0.0001)
        .test(samples, 100000, 0.01, 10);
  }
  
  @Ignore
  @Test
  public void test_BasicNN_XOR_Softmax() throws Exception {
    final int[] midSize = new int[] { 2 };
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 0, 1 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 1, 0 }) },
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1, 0 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 0, 1 }) }
    };
    new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize).addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian()))
        .add(new SoftmaxActivationLayer())
        .setVerbose(true)
        .test(samples, 100000, 0.01, 10);
  }
  
  @Test
  public void test_BasicNN_OR() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] midSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        // XOR:
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { -1 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1 }) }
    };
    new PipelineNetwork()
        // Becomes unstable if these are added:
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize).addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian()))
        .add(new BiasLayer(midSize))
        .add(new SigmoidActivationLayer())
        
        // Works okay:
        .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize).addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian()))
        .add(new BiasLayer(outSize))
        .add(new SigmoidActivationLayer())
        .setRate(0.001)
        .test(samples, 100000, 0.01, 10);
  }
  
  @Test
  public void test_BasicNN_XOR() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] midSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        // XOR:
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { -1 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { -1 }) }
    };
    new PipelineNetwork()
        
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize)
            .addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian())
            .setMomentumDecay(0.9)
            .setMass(5.))
        .add(new BiasLayer(midSize).setMomentumDecay(0.5).setMass(2.))
        .add(new SigmoidActivationLayer())
        
        .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize)
            .addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian())
            .setMomentumDecay(0.5))
        .add(new BiasLayer(outSize))
        .add(new SigmoidActivationLayer())
        
        .setRate(0.0001).setMutationAmount(0.1)
        .test(samples, 10000, 0.01, 10);
  }
  
  @Test
  public void test_BasicNN_XOR_3layer() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] midSize = new int[] { 4 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        // XOR:
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { -1 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { -1 }) }
    };
    new PipelineNetwork()
        .add(
            new DenseSynapseLayer(NDArray.dim(inputSize), midSize).addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian()).setMass(5.)
                .setMomentumDecay(0.5))
        .add(new BiasLayer(midSize).setMass(5.).setMomentumDecay(0.5))
        .add(new SigmoidActivationLayer())
        .add(
            new DenseSynapseLayer(NDArray.dim(midSize), midSize).addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian()).setMomentumDecay(0.8)
                .setMass(2.))
        .add(new BiasLayer(midSize).setMomentumDecay(0.8).setMass(2.))
        .add(new SigmoidActivationLayer())
        .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize).addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian()).setMomentumDecay(0.9))
        .add(new BiasLayer(outSize).setMomentumDecay(0.9))
        .add(new SigmoidActivationLayer())
        .setRate(0.0001)//.setVerbose(true)
        .test(samples, 100000, 0.01, 10);
  }
  
  @Test
  public void test_LinearNN() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 0.5 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 0.5 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 0 }) }
    };
    new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian()))
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian()))
        .setRate(0.25)
        .test(samples, 10000, 0.1, 100);
  }
  
  @Test
  public void testDenseLinearLayer_2Layer() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1, 0 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0, 1 }) }
    };
    
    new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian()))
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian()))
        .test(samples, 10000, 0.01, 10);
    
    new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian()).freeze())
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian()))
        .test(samples, 10000, 0.01, 10);
    
    new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian()))
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian()).freeze())
        .test(samples, 10000, 0.01, 10);
  }
  
  @Test
  public void testDenseLinearLayer_Basic() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1, 0 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0, 1 }) }
    };
    
    new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * ImageNetworkDev.random.nextGaussian()))
        .setRate(0.1)
        .test(samples, 10000, 0.01, 100);
  }
  
}
