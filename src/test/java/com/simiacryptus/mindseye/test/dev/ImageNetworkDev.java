package com.simiacryptus.mindseye.test.dev;

import java.awt.Desktop;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import javax.imageio.ImageIO;

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
    
    //NDArray inputImage = TestMNISTDev.toNDArray3(scale(ImageIO.read(getClass().getResourceAsStream("/monkey1.jpg")),.5));
    NDArray inputImage = TestMNISTDev.toNDArray1(render(new int[]{240,200}, "Hello World"));
    //NDArray inputImage = TestMNISTDev.toNDArray3(render(new int[]{300,300}, "Hello World"));
    
    
    final int[] inputSize = inputImage.getDims();
    int[] kernelSize = new int[] { 5, 5, 1 };
    int[] kernelSize2 = new int[] { 2, 2, 1 };
    final int[] outSize = outsize(inputSize, kernelSize);
    final int[] outSize2 = outsize(inputSize, kernelSize2);
    
    // List<LabeledObject<NDArray>> data = TestMNISTDev.trainingDataStream().limit(10).collect(Collectors.toList());
    List<LabeledObject<NDArray>> data = new ArrayList<>();
    data.add(new LabeledObject<NDArray>(inputImage, ""));

    ConvolutionSynapseLayer convolution = new ConvolutionSynapseLayer(kernelSize, 1);
    convolution.kernel.set(new int[] { 0, 4, 0, 0 }, 0.125);
    convolution.kernel.set(new int[] { 1, 3, 0, 0 }, 0.25);
    convolution.kernel.set(new int[] { 2, 2, 0, 0 }, .25);
    //convolution.kernel.set(new int[] { 3, 3, 0, 0 }, 1);
    convolution.kernel.set(new int[] { 3, 1, 0, 0 }, 0.25);
    convolution.kernel.set(new int[] { 4, 0, 0, 0 }, 0.125);
    convolution.freeze();

    ConvolutionSynapseLayer convolution2 = new ConvolutionSynapseLayer(kernelSize2, 1);
    convolution2.kernel.set(new int[] { 0, 0, 0, 0 }, 1);
    convolution2.kernel.set(new int[] { 1, 0, 0, 0 }, -1);
    convolution2.kernel.set(new int[] { 0, 1, 0, 0 }, -1);
    convolution2.kernel.set(new int[] { 1, 1, 0, 0 }, 1);
    convolution2.freeze();
    
    PipelineNetwork forwardConvolutionNet = new PipelineNetwork().add(convolution);
    
    report(data.stream().map(obj -> {
      NNResult output = forwardConvolutionNet.eval(obj.data);
      NDArray zeroInput = new NDArray(inputSize);
      NDArray zeroOutput = new NDArray(outSize);
      NDArray zeroOutput2 = new NDArray(outSize2);
      BiasLayer bias = new BiasLayer(inputSize);//.setHalflife(2);//.setSampling(0.8);
      Trainer trainer = new Trainer();
      
      // convolution.setVerbose(true);
      trainer.add(new PipelineNetwork()
          .add(bias)
          .add(convolution),
          new NDArray[][] { { zeroInput, output.data } });
      
//      trainer.add(new SupervisedTrainingParameters(new PipelineNetwork()
//          .add(bias)
//          .add(convolution2), new NDArray[][] { { zeroInput, zeroOutput2 } })
//          .setWeight(2.));
      
      
      
//      //
//      trainer.add(new SupervisedTrainingParameters(
//          new PipelineNetwork().add(bias),
//          new NDArray[][] { { zeroInput, zeroInput } })
//          .setWeight(1));
      
      trainer
          .setMutationAmount(0.0)
          // .setImprovementStaleThreshold(Integer.MAX_VALUE)
          .setStaticRate(50.)
          .setVerbose(true)
          .setLoopA(1)
          .setLoopB(1)
          .setRateAdaptionRate(0.3)
          .setDynamicRate(0.02)
          .setMaxDynamicRate(0.5)
          .train(2, 0.0001);

      bias = (BiasLayer) trainer.getBest().getFirst().get(0).getNet().get(0);
      NNResult recovered = bias.eval(zeroInput);
      NNResult tested = new PipelineNetwork().add(bias).add(convolution).eval(zeroInput);
      
      return imageHtml(
          TestMNISTDev.toImage(obj.data),
          TestMNISTDev.toImage(new NDArray(outSize, output.data.data)),
          TestMNISTDev.toImage(new NDArray(inputSize, recovered.data.data)),
          TestMNISTDev.toImage(new NDArray(outSize, tested.data.data))
      );
    }));
    
  }

  public int[] outsize(final int[] inputSize, int[] kernelSize) {
    final int[] outSize = new int[] { inputSize[0] - kernelSize[0] + 1, inputSize[1] - kernelSize[1] + 1, inputSize[2] - kernelSize[2] + 1 };
    return outSize;
  }

  public static String imageHtml(BufferedImage... imgArray) {
    return Stream.of(imgArray).map(img -> TestMNISTDev.toInlineImage(img, "")).reduce((a, b) -> a + b).get();
  }

  public static void report(String... fragments) throws FileNotFoundException, IOException {
    report(Stream.of(fragments));
  }

  public static void report(Stream<String> fragments) throws FileNotFoundException, IOException {
    final File outDir = new File("reports");
    outDir.mkdirs();
    final StackTraceElement caller = Thread.currentThread().getStackTrace()[2];
    final File report = new File(outDir, caller.getClassName() + "_" + caller.getLineNumber() + ".html");
    final PrintStream out = new PrintStream(new FileOutputStream(report));
    out.println("<html><head></head><body>");
    fragments.forEach(out::println);
    out.println("</body></html>");
    out.close();
    Desktop.getDesktop().browse(report.toURI());
  }

  public static BufferedImage scale(BufferedImage img, double scale) {
    int w = img.getWidth();
    int h = img.getHeight();
    BufferedImage after = new BufferedImage((int)(w*scale), (int)(h*scale), BufferedImage.TYPE_INT_ARGB);
    AffineTransform at = new AffineTransform();
    at.scale(scale, scale);
    AffineTransformOp scaleOp = 
       new AffineTransformOp(at, AffineTransformOp.TYPE_BILINEAR);
    img = scaleOp.filter(img, after);
    return img;
  }
  
  private BufferedImage render(final int[] inputSize, String string) {
    Random r = new Random();
    BufferedImage img = new BufferedImage(inputSize[0], inputSize[1], BufferedImage.TYPE_INT_RGB);
    Graphics2D g = img.createGraphics();
    for(int i=0;i<20;i++)
    {
      int size = (int) (24 + 32 * r.nextGaussian());
      int x = (int) ((inputSize[0]/2) * (1+r.nextGaussian()));
      int y = (int) ((inputSize[1]/2) * (1+r.nextGaussian()));
      g.setFont(g.getFont().deriveFont(Font.PLAIN, size));
      g.drawString(string, x, y);
    }
    return img;
  }
  
}
