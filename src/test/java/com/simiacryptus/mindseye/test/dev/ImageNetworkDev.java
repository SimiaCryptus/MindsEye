package com.simiacryptus.mindseye.test.dev;

import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.data.LabeledObject;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.PipelineNetwork;
import com.simiacryptus.mindseye.training.SupervisedTrainingParameters;
import com.simiacryptus.mindseye.training.Trainer;

@SuppressWarnings("unused")
public class ImageNetworkDev {
  static final Logger log = LoggerFactory.getLogger(ImageNetworkDev.class);
  
  public static final Random random = new Random();
  
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
    return new int[] { inputSize[0] - kernelSize[0] + 1, inputSize[1] - kernelSize[1] + 1, inputSize[2] - kernelSize[2] + 1 };
  }
  
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
    // final NDArray inputImage = TestMNISTDev.toNDArray3(ImageNetworkDev.scale(ImageIO.read(getClass().getResourceAsStream("/monkey1.jpg")), .5));
    NDArray inputImage = TestMNISTDev.toNDArray1(render(new int[] { 200, 200 }, "Hello World"));
    // NDArray inputImage = TestMNISTDev.toNDArray3(render(new int[]{300,300}, "Hello World"));
    
    final NNLayer convolution = blur_3x4();
    
    final int[] inputSize = inputImage.getDims();
    final int[] outSize = convolution.eval(new NDArray(inputSize)).data.getDims();
    final List<LabeledObject<NDArray>> data = new ArrayList<>();
    data.add(new LabeledObject<NDArray>(inputImage, ""));
    
    final PipelineNetwork forwardConvolutionNet = new PipelineNetwork().add(convolution);
    
    Util.report(data.stream().map(obj -> {
      final NNResult output = forwardConvolutionNet.eval(obj.data);
      final NDArray zeroInput = new NDArray(inputSize);
      BiasLayer bias = new BiasLayer(inputSize);
      final Trainer trainer = new Trainer().setStaticRate(1.);
      
      trainer.add(new SupervisedTrainingParameters(new PipelineNetwork()
          .add(bias)
          .add(convolution), new NDArray[][] { { zeroInput, output.data } }).setWeight(1));
      
      // trainer.add(new SupervisedTrainingParameters(
      // new PipelineNetwork().add(bias),
      // new NDArray[][] { { zeroInput, zeroInput } })
      // {
      // @Override
      // public NDArray getIdeal(NNResult eval, NDArray preset) {
      // NDArray retVal = preset.copy();
      // for (int i = 0; i < retVal.dim(); i++) {
      // double x = eval.data.getData()[i];
      // retVal.getData()[i] = ((x > -0.1)?x:0)*0.99;
      // }
      // return retVal;
      // }
      // }.setWeight(1));
        
        trainer.add(new SupervisedTrainingParameters(
            new PipelineNetwork().add(bias).add(new com.simiacryptus.mindseye.layers.MaxEntLayer()),
            new NDArray[][] { { zeroInput, new NDArray(1) } }).setWeight(0.1));

        // trainer.add(new SupervisedTrainingParameters(
        // new PipelineNetwork().add(bias).add(new com.simiacryptus.mindseye.layers.MaxEntLayer().setFactor(1).setReverse(true)),
        // new NDArray[][] { { zeroInput, new NDArray(1) } }).setWeight(-0.1));
        
        trainer
            .setMutationAmount(0.1)
            // .setStaticRate(0.05)
            .setVerbose(true)
            // .setDynamicRate(0.005)
            // .setMaxDynamicRate(1.)
            // .setMinDynamicRate(0.001)
            .train(0, 0.1);
        
        bias = (BiasLayer) trainer.getBest().getFirst().get(0).getNet().get(0);
        final NNResult recovered = bias.eval(zeroInput);
        final NNResult tested = new PipelineNetwork().add(bias).add(convolution).eval(zeroInput);
        
        return Util.imageHtml(
            Util.toImage(obj.data),
            Util.toImage(new NDArray(outSize, output.data.getData())),
            Util.toImage(new NDArray(inputSize, recovered.data.getData())),
            Util.toImage(new NDArray(outSize, tested.data.getData()))
            );
      }));
    
  }
  
}
