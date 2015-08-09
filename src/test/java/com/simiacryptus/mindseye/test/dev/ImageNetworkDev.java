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
import com.simiacryptus.mindseye.data.LabeledObject;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.training.PipelineNetwork;
import com.simiacryptus.mindseye.training.SupervisedTrainingParameters;
import com.simiacryptus.mindseye.training.Trainer;

public class ImageNetworkDev {
  static final Logger log = LoggerFactory.getLogger(ImageNetworkDev.class);
  
  public static final Random random = new Random();
  
  public static String imageHtml(final BufferedImage... imgArray) {
    return Stream.of(imgArray).map(img -> TestMNISTDev.toInlineImage(img, "")).reduce((a, b) -> a + b).get();
  }
  
  public static void report(final Stream<String> fragments) throws FileNotFoundException, IOException {
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
  
  public static void report(final String... fragments) throws FileNotFoundException, IOException {
    ImageNetworkDev.report(Stream.of(fragments));
  }
  
  public static BufferedImage scale(BufferedImage img, final double scale) {
    final int w = img.getWidth();
    final int h = img.getHeight();
    final BufferedImage after = new BufferedImage((int) (w * scale), (int) (h * scale), BufferedImage.TYPE_INT_ARGB);
    final AffineTransform at = new AffineTransform();
    at.scale(scale, scale);
    final AffineTransformOp scaleOp =
        new AffineTransformOp(at, AffineTransformOp.TYPE_BILINEAR);
    img = scaleOp.filter(img, after);
    return img;
  }
  
  public NNLayer blur_3() {
    final ConvolutionSynapseLayer convolution = new ConvolutionSynapseLayer(new int[] { 3, 3, 1 }, 1);
    convolution.kernel.set(new int[] { 0, 0, 0, 0 }, 0.333);
    convolution.kernel.set(new int[] { 0, 1, 0, 0 }, 0);
    convolution.kernel.set(new int[] { 0, 2, 0, 0 }, 0);
    convolution.kernel.set(new int[] { 1, 0, 0, 0 }, 0.);
    convolution.kernel.set(new int[] { 1, 1, 0, 0 }, 0.333);
    convolution.kernel.set(new int[] { 1, 2, 0, 0 }, 0.);
    convolution.kernel.set(new int[] { 2, 0, 0, 0 }, 0.);
    convolution.kernel.set(new int[] { 2, 1, 0, 0 }, 0.);
    convolution.kernel.set(new int[] { 2, 2, 0, 0 }, 0.333);
    convolution.freeze();
    return convolution;
  }
  
  public NNLayer blur_3x4() {
    final PipelineNetwork net = new PipelineNetwork();
    for (int i = 0; i < 3; i++)
    {
      net.add(blur_3());
    }
    return net;
  }
  
  public NNLayer blur1() {
    final ConvolutionSynapseLayer convolution2 = new ConvolutionSynapseLayer(new int[] { 2, 2, 1 }, 1);
    convolution2.kernel.set(new int[] { 0, 0, 0, 0 }, 0.25);
    convolution2.kernel.set(new int[] { 1, 0, 0, 0 }, 0.25);
    convolution2.kernel.set(new int[] { 0, 1, 0, 0 }, 0.25);
    convolution2.kernel.set(new int[] { 1, 1, 0, 0 }, 0.25);
    convolution2.freeze();
    return convolution2;
  }
  
  public NNLayer edge1() {
    final ConvolutionSynapseLayer convolution2 = new ConvolutionSynapseLayer(new int[] { 2, 2, 1 }, 1);
    convolution2.kernel.set(new int[] { 0, 0, 0, 0 }, -1);
    convolution2.kernel.set(new int[] { 1, 0, 0, 0 }, 1);
    convolution2.kernel.set(new int[] { 0, 1, 0, 0 }, 1);
    convolution2.kernel.set(new int[] { 1, 1, 0, 0 }, -1);
    convolution2.freeze();
    return convolution2;
  }
  
  public int[] outsize(final int[] inputSize, final int[] kernelSize) {
    final int[] outSize = new int[] { inputSize[0] - kernelSize[0] + 1, inputSize[1] - kernelSize[1] + 1, inputSize[2] - kernelSize[2] + 1 };
    return outSize;
  }
  
  @SuppressWarnings("unused")
  private BufferedImage render(final int[] inputSize, final String string) {
    final Random r = new Random();
    final BufferedImage img = new BufferedImage(inputSize[0], inputSize[1], BufferedImage.TYPE_INT_RGB);
    final Graphics2D g = img.createGraphics();
    for (int i = 0; i < 20; i++)
    {
      final int size = (int) (24 + 32 * r.nextGaussian());
      final int x = (int) (inputSize[0] / 2 * (1 + r.nextGaussian()));
      final int y = (int) (inputSize[1] / 2 * (1 + r.nextGaussian()));
      g.setFont(g.getFont().deriveFont(Font.PLAIN, size));
      g.drawString(string, x, y);
    }
    return img;
  }
  
  @Test
  public void testDeconvolution() throws Exception {
    
    // List<LabeledObject<NDArray>> data = TestMNISTDev.trainingDataStream().limit(10).collect(Collectors.toList());
    final NDArray inputImage = TestMNISTDev.toNDArray3(ImageNetworkDev.scale(ImageIO.read(getClass().getResourceAsStream("/monkey1.jpg")), .5));
    // NDArray inputImage = TestMNISTDev.toNDArray1(render(new int[] { 200, 200 }, "Hello World"));
    // NDArray inputImage = TestMNISTDev.toNDArray3(render(new int[]{300,300}, "Hello World"));
    
    final NNLayer convolution = blur_3x4();
    
    final int[] inputSize = inputImage.getDims();
    final int[] outSize = convolution.eval(new NDArray(inputSize)).data.getDims();
    final List<LabeledObject<NDArray>> data = new ArrayList<>();
    data.add(new LabeledObject<NDArray>(inputImage, ""));
    
    final PipelineNetwork forwardConvolutionNet = new PipelineNetwork().add(convolution);
    
    ImageNetworkDev.report(data.stream().map(obj -> {
      final NNResult output = forwardConvolutionNet.eval(obj.data);
      final NDArray zeroInput = new NDArray(inputSize);
      BiasLayer bias = new BiasLayer(inputSize);
      final Trainer trainer = new Trainer();
      
      trainer.add(new SupervisedTrainingParameters(new PipelineNetwork()
          .add(bias)
          .add(convolution), new NDArray[][] { { zeroInput, output.data } }).setWeight(1));
      
      trainer.add(new SupervisedTrainingParameters(
          new PipelineNetwork().add(bias),
          new NDArray[][] { { zeroInput, zeroInput } })
      {
        @Override
        public NDArray getIdeal(NNResult eval, NDArray preset) {
          NDArray retVal = preset.copy();
          for (int i = 0; i < retVal.dim(); i++) {
            if (eval.data.getData()[i] < 0) retVal.getData()[i] = eval.data.getData()[i];
          }
          return retVal;
        }
      }.setWeight(1));
      
//      trainer.add(new SupervisedTrainingParameters(
//          new PipelineNetwork().add(bias).add(new com.simiacryptus.mindseye.layers.MaxEntLayer().setFactor(5)),
//          new NDArray[][] { { zeroInput, new NDArray(1) } }).setWeight(1));
      
      trainer
          .setMutationAmount(0.1)
          //.setStaticRate(0.05)
          .setVerbose(true)
          //.setDynamicRate(0.005)
          // .setMaxDynamicRate(1.)
          // .setMinDynamicRate(0.001)
          .train(100, 0.0001);
      
      bias = (BiasLayer) trainer.getBest().getFirst().get(0).getNet().get(0);
      final NNResult recovered = bias.eval(zeroInput);
      final NNResult tested = new PipelineNetwork().add(bias).add(convolution).eval(zeroInput);
      
      return ImageNetworkDev.imageHtml(
          TestMNISTDev.toImage(obj.data),
          TestMNISTDev.toImage(new NDArray(outSize, output.data.getData())),
          TestMNISTDev.toImage(new NDArray(inputSize, recovered.data.getData())),
          TestMNISTDev.toImage(new NDArray(outSize, tested.data.getData()))
          );
    }));
    
  }
  
}
