package com.simiacryptus.mindseye.test.dev;

import java.awt.Desktop;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.PipelineNetwork;
import com.simiacryptus.mindseye.SupervisedTrainingParameters;
import com.simiacryptus.mindseye.Trainer;
import com.simiacryptus.mindseye.data.LabeledObject;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.learning.NNResult;

public class ImageNetworkDev {
  static final Logger log = LoggerFactory.getLogger(ImageNetworkDev.class);
  
  public static final Random random = new Random();
  
  @Test
  public void testDeconvolution() throws Exception {
    
    final int[] inputSize = new int[] { 700, 200 };
    int[] kernelSize = new int[] { 5, 5 };
    final int[] outSize = new int[] { inputSize[0] - kernelSize[0] + 1, inputSize[1] - kernelSize[1] + 1 };
    
    // List<LabeledObject<NDArray>> data = TestMNISTDev.trainingDataStream().limit(10).collect(Collectors.toList());
    List<LabeledObject<NDArray>> data = new ArrayList<>();
    data.add(new LabeledObject<NDArray>(TestMNISTDev.toNDArray(render(inputSize, "Hello World")), ""));
    
    ConvolutionSynapseLayer convolution = new ConvolutionSynapseLayer(kernelSize, 1);
    convolution.kernel.set(new int[] { 0, 4, 0 }, 0.5);
    convolution.kernel.set(new int[] { 1, 3, 0 }, 0.75);
    convolution.kernel.set(new int[] { 2, 2, 0 }, 1);
    convolution.kernel.set(new int[] { 3, 1, 0 }, 0.75);
    convolution.kernel.set(new int[] { 4, 0, 0 }, 0.5);
    convolution.freeze();
    
    PipelineNetwork forwardConvolutionNet = new PipelineNetwork().add(convolution);
    
    Stream<BufferedImage[]> buffer = data.stream().map(obj -> {
      NNResult output = forwardConvolutionNet.eval(obj.data);
      NDArray zero = new NDArray(inputSize);
      BiasLayer bias = new BiasLayer(inputSize).setSampling(0.01);
      Trainer trainer = new Trainer();
      
      // convolution.setVerbose(true);
      trainer.add(new PipelineNetwork()
          .add(bias)
          .add(convolution),
          new NDArray[][] { { obj.data, obj.data } });
      
      trainer.add(new SupervisedTrainingParameters(
          new PipelineNetwork().add(bias),
          new NDArray[][] { { zero, zero } })
          .setWeight(10));
      
      trainer.setMutationAmount(0.0)
          // .setImprovementStaleThreshold(Integer.MAX_VALUE)
          .setRate(1.)
          .setVerbose(true)
          .train(100, 0.01);
      
      bias = (BiasLayer) trainer.getBest().getFirst().get(0).getNet().get(0);
      NNResult recovered = bias.eval(obj.data);
      NNResult tested = forwardConvolutionNet.eval(recovered.data);
      
      return new BufferedImage[] {
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
    buffer.map(x -> "<p>" + Stream.of(x).map(o -> TestMNISTDev.toInlineImage(o, "")).reduce((a, b) -> a + b).get() + "</p>")
        .forEach(out::println);
    out.println("</body></html>");
    out.close();
    Desktop.getDesktop().browse(report.toURI());
    
  }
  
  private BufferedImage render(final int[] inputSize, String string) {
    Random r = new Random();
    BufferedImage img = new BufferedImage(inputSize[0], inputSize[1], BufferedImage.TYPE_INT_RGB);
    Graphics2D g = img.createGraphics();
    for(int i=0;i<20;i++)
    {
      int size = (int) (24 + 32 * r.nextGaussian());
      int x = (int) (250 + 350 * r.nextGaussian());
      int y = (int) (130 + 130 * r.nextGaussian());
      g.setFont(g.getFont().deriveFont(Font.PLAIN, size));
      g.drawString(string, x, y);
    }
    return img;
  }
  
}
