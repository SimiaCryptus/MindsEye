package com.simiacryptus.mindseye.test.demo;

import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.LabeledObject;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNResult;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.DAGNetwork.DAGNode;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.basic.SqActivationLayer;
import com.simiacryptus.mindseye.net.basic.SqLossLayer;
import com.simiacryptus.mindseye.net.basic.SumLayer;
import com.simiacryptus.mindseye.net.basic.VerboseWrapper;
import com.simiacryptus.mindseye.net.dev.LinearActivationLayer;
import com.simiacryptus.mindseye.net.dev.ThresholdActivationLayer;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.media.MaxConstLayer;
import com.simiacryptus.mindseye.test.Tester;
import com.simiacryptus.mindseye.training.GradientDescentTrainer;
import com.simiacryptus.mindseye.training.TrainingContext;

public class DeconvolutionTest {
  static final Logger log = LoggerFactory.getLogger(DeconvolutionTest.class);

  public static final Random random = new Random();

  public static BufferedImage render(final int[] inputSize, final String string) {
    final Random r = new Random();
    final BufferedImage img = new BufferedImage(inputSize[0], inputSize[1], BufferedImage.TYPE_INT_RGB);
    final Graphics2D g = img.createGraphics();
    for (int i = 0; i < 20; i++) {
      final int size = (int) (24 + 32 * r.nextGaussian());
      final int x = (int) (inputSize[0] / 2 * (1 + r.nextGaussian()));
      final int y = (int) (inputSize[1] / 2 * (1 + r.nextGaussian()));
      g.setFont(g.getFont().deriveFont(Font.PLAIN, size));
      g.drawString(string, x, y);
    }
    return img;
  }

  public static BufferedImage scale(BufferedImage img, final double scale) {
    final int w = img.getWidth();
    final int h = img.getHeight();
    final BufferedImage after = new BufferedImage((int) (w * scale), (int) (h * scale), BufferedImage.TYPE_INT_ARGB);
    final AffineTransform at = new AffineTransform();
    at.scale(scale, scale);
    final AffineTransformOp scaleOp = new AffineTransformOp(at, AffineTransformOp.TYPE_BILINEAR);
    img = scaleOp.filter(img, after);
    return img;
  }

  public NNLayer<?> blur_3() {
    final ConvolutionSynapseLayer convolution = new ConvolutionSynapseLayer(new int[] { 3, 3 }, 9);
    for(int ii=0;ii<3;ii++){
      int i = ii+ii*3;
      convolution.kernel.set(new int[] { 0, 2, i }, 0.333);
      convolution.kernel.set(new int[] { 1, 1, i }, 0.333);
      convolution.kernel.set(new int[] { 2, 0, i }, 0.333);
    }
    convolution.freeze();
    return convolution;
  }

  public NNLayer<?> blur_3x4() {
    final DAGNetwork net = new DAGNetwork();
    for (int i = 0; i < 3; i++) {
      net.add(blur_3());
    }
    return net;
  }

  public int[] outsize(final int[] inputSize, final int[] kernelSize) {
    return new int[] { inputSize[0] - kernelSize[0] + 1, inputSize[1] - kernelSize[1] + 1, inputSize[2] - kernelSize[2] + 1 };
  }

  @Test
  public void testDeconvolution() throws Exception {

    // List<LabeledObject<NDArray>> data = TestMNISTDev.trainingDataStream().limit(10).collect(Collectors.toList());
    final NDArray inputImage = Util.toNDArray3(DeconvolutionTest.scale(ImageIO.read(getClass().getResourceAsStream("/monkey1.jpg")), .5));
    //final NDArray inputImage = Util.toNDArray1(render(new int[] { 200, 300 }, "Hello World"));
    //NDArray inputImage = Util.toNDArray3(render(new int[]{200,300}, "Hello World"));

    final NNLayer<?> convolution = blur_3x4();
    //final NNLayer<?> convolution = blur_3();
    
    final int[] inputSize = inputImage.getDims();
    final int[] outSize = convolution.eval(new NDArray(inputSize)).data.getDims();
    final List<LabeledObject<NDArray>> data = new ArrayList<>();
    data.add(new LabeledObject<NDArray>(inputImage, "Ideal Input"));

    Util.report(data.stream().map(obj -> {

      final NNResult blurredImage = convolution.eval(new NDArray[] { obj.data });
      
      final NDArray zeroInput = new NDArray(inputSize);
      final DAGNetwork dagNetwork = new DAGNetwork();
      BiasLayer bias = new BiasLayer(inputSize){

        @Override
        public double[] add(double[] input) {
          final double[] array = new double[input.length];
          for (int i = 0; i < array.length; i++) {
            double v = input[i] + this.bias[i];
            array[i] = v<0?0:v;
          }
          return array;
        }
        
      };
      dagNetwork.add(bias);
      DAGNode modeledImageNode = dagNetwork.getHead();
      
      dagNetwork.add(convolution);
      dagNetwork.addLossComponent(new SqLossLayer());
      //dagNetwork.add(new BiasLayer(new int[]{1}).setWeights(i->.5).freeze());
      //dagNetwork.add(new MaxConstLayer().setValue(10));

      List<DAGNode> outs = new ArrayList<>();
      outs.add(dagNetwork.getHead());

      new ThresholdActivationLayer();

      final LinearActivationLayer edgeGateH;
      {
        final ConvolutionSynapseLayer edgeFilter = new ConvolutionSynapseLayer(new int[] { 1, 2 }, 9);
        for(int ii=0;ii<3;ii++){
          int i = ii+ii*3;
          edgeFilter.kernel.set(new int[] { 0, 0, i }, -1);
          edgeFilter.kernel.set(new int[] { 0, 1, i }, 1);
        }
        edgeFilter.freeze();
        dagNetwork.add(edgeFilter, modeledImageNode);
        
        dagNetwork.add(new com.simiacryptus.mindseye.net.dev.L1SimpleNormalizationLayer());
        dagNetwork.add(new com.simiacryptus.mindseye.net.media.MaxEntLayer());
        dagNetwork.add(new SigmoidActivationLayer().setBalanced(false));
        dagNetwork.add(new SumLayer());
        
        edgeGateH = new LinearActivationLayer().setWeights(new double[]{0.}).freeze();
        dagNetwork.add(edgeGateH);
        // Add 1 to output so product stays above 0 since this fitness function is secondary
        dagNetwork.add(new BiasLayer(new int[]{1}).setWeights(i->1).freeze());
        outs.add(dagNetwork.getHead());
      }

      final LinearActivationLayer edgeGateV;
      {
        final ConvolutionSynapseLayer edgeFilter = new ConvolutionSynapseLayer(new int[] { 2, 1 }, 9);
        for(int ii=0;ii<3;ii++){
          int i = ii+ii*3;
          edgeFilter.kernel.set(new int[] { 0, 0, i }, -1);
          edgeFilter.kernel.set(new int[] { 1, 0, i }, 1);
        }
        edgeFilter.freeze();
        dagNetwork.add(edgeFilter, modeledImageNode);
        
        dagNetwork.add(new com.simiacryptus.mindseye.net.dev.L1SimpleNormalizationLayer());
        dagNetwork.add(new com.simiacryptus.mindseye.net.media.MaxEntLayer());
        dagNetwork.add(new SigmoidActivationLayer().setBalanced(false));
        dagNetwork.add(new SumLayer());
        
        edgeGateV = new LinearActivationLayer().setWeights(new double[]{0.}).freeze();
        dagNetwork.add(edgeGateV);
        // Add 1 to output so product stays above 0 since this fitness function is secondary
        dagNetwork.add(new BiasLayer(new int[]{1}).setWeights(i->1).freeze());
        outs.add(dagNetwork.getHead());
      }

      dagNetwork.add(new VerboseWrapper("endprod", new com.simiacryptus.mindseye.net.dev.StrangeProductLayer()), outs.toArray(new DAGNode[]{}));
      

      final Tester trainer = new Tester().setStaticRate(1.);
      
      //new NetInitializer().initialize(initPredictionNetwork);
      GradientDescentTrainer gradientDescentTrainer = trainer.getGradientDescentTrainer();
      gradientDescentTrainer.setNet(dagNetwork);
      gradientDescentTrainer.setData(new NDArray[][] { { zeroInput, blurredImage.data } });
      final TrainingContext trainingContext = new TrainingContext().setTimeout(3, java.util.concurrent.TimeUnit.MINUTES);
      try {
        trainer.setStaticRate(0.5).setMaxDynamicRate(1000000).setVerbose(true);
        trainer.train(15., trainingContext);
        trainer.getDevtrainer().reset();
        //edgeGateH.setWeights(new double[]{1.});
        edgeGateV.setWeights(new double[]{1.});
        //negativeClamp.setFactor(-1);
        trainer.train(0.0, trainingContext);        
        trainer.getDevtrainer().reset();
      } catch (final Exception e) {
        e.printStackTrace();
      }

      bias = (BiasLayer) trainer.getNet().getChild(bias.getId());
      final NNResult recovered = bias.eval(zeroInput);
      final NNResult verification = new DAGNetwork().add(bias).add(convolution).eval(zeroInput);

      return Util.imageHtml( //
          Util.toImage(obj.data), //
          Util.toImage(new NDArray(outSize, blurredImage.data.getData())), // 
          Util.toImage(new NDArray(inputSize, recovered.data.getData())), //
          Util.toImage(new NDArray(outSize, verification.data.getData())));
    }));

  }
  
  @Test
  public void testDeconvolution2() throws Exception {

    // List<LabeledObject<NDArray>> data =
    // TestMNISTDev.trainingDataStream().limit(10).collect(Collectors.toList());
    final NDArray inputImage = Util.toNDArray3(DeconvolutionTest.scale(ImageIO.read(getClass().getResourceAsStream("/monkey1.jpg")), .5));
    //final NDArray inputImage = Util.toNDArray1(render(new int[] { 200, 200 }, "Hello World"));
//     NDArray inputImage = TestMNISTDev.toNDArray3(render(new int[]{300,300}, "Hello World"));

    final NNLayer<?> convolution = blur_3x4();

    final int[] inputSize = inputImage.getDims();
    final int[] outSize = convolution.eval(new NDArray(inputSize)).data.getDims();
    final List<LabeledObject<NDArray>> data = new ArrayList<>();
    data.add(new LabeledObject<NDArray>(inputImage, "Ideal Input"));

    final DAGNetwork forwardConvolutionNet = new DAGNetwork().add(convolution);

    Util.report(data.stream().map(obj -> {
      final NDArray[] input = { obj.data };
      final NNResult output = forwardConvolutionNet.eval(input);
      final NDArray zeroInput = new NDArray(inputSize);
      BiasLayer bias = new BiasLayer(inputSize);
      final Tester trainer = new Tester().setStaticRate(1.);

      trainer.init(new NDArray[][] { { zeroInput, output.data } }, new DAGNetwork().add(bias).add(convolution), new SqLossLayer());

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

      // trainer.add(new SupervisedTrainingParameters(
      // new PipelineNetwork().add(bias).add(new
      // com.simiacryptus.mindseye.layers.MaxEntLayer()),
      // new NDArray[][] { { zeroInput, new NDArray(1) } }).setWeight(0.1));

      // trainer.add(new SupervisedTrainingParameters(
      // new PipelineNetwork().add(bias).add(new
      // com.simiacryptus.mindseye.layers.MaxEntLayer().setFactor(1).setReverse(true)),
      // new NDArray[][] { { zeroInput, new NDArray(1) } }).setWeight(-0.1));

      final TrainingContext trainingContext = new TrainingContext().setTimeout(15, java.util.concurrent.TimeUnit.MINUTES);
      try {
        trainer.setStaticRate(0.5).setMaxDynamicRate(1000000).setVerbose(true).train(0.0, trainingContext);
        trainer.getDevtrainer().reset();
      } catch (final Exception e) {
        e.printStackTrace();
      }

      bias = (BiasLayer) trainer.getNet().getChild(bias.getId());
      final NNResult recovered = bias.eval(zeroInput);
      final NNResult tested = new DAGNetwork().add(bias).add(convolution).eval(zeroInput);

      return Util.imageHtml(Util.toImage(obj.data), Util.toImage(new NDArray(outSize, output.data.getData())), Util.toImage(new NDArray(inputSize, recovered.data.getData())),
          Util.toImage(new NDArray(outSize, tested.data.getData())));
    }));

  }

  @Test
  public void testConvolution() throws Exception {

    final NDArray inputImage = Util.toNDArray3(DeconvolutionTest.scale(ImageIO.read(getClass().getResourceAsStream("/monkey1.jpg")), .5));
    //final NDArray inputImage = Util.toNDArray1(render(new int[] { 200, 300 }, "Hello World"));

    final NNLayer<?> convolution = blur_3x4();
    //final NNLayer<?> convolution = blur_3();

    final int[] inputSize = inputImage.getDims();
    final int[] outSize = convolution.eval(new NDArray(inputSize)).data.getDims();
    final List<LabeledObject<NDArray>> data = new ArrayList<>();
    data.add(new LabeledObject<NDArray>(inputImage, "Ideal Input"));

    final DAGNetwork forwardConvolutionNet = new DAGNetwork().add(convolution);

    Util.report(data.stream().map(obj -> {
      final NDArray[] input = { obj.data };
      final NNResult output = forwardConvolutionNet.eval(input);

      return Util.imageHtml(
          Util.toImage(obj.data), 
          Util.toImage(new NDArray(outSize, output.data.getData()))
        );
    }));

  }

}
