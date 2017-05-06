package com.simiacryptus.mindseye.test.demo;

import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import javax.imageio.ImageIO;

import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.net.SupervisedNetwork;
import com.simiacryptus.util.ml.Tensor;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.util.Util;
import com.simiacryptus.util.test.LabeledObject;
import com.simiacryptus.mindseye.training.TrainingContext;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;
import com.simiacryptus.mindseye.net.PipelineNetwork;
import com.simiacryptus.mindseye.net.dag.DAGNode;
import com.simiacryptus.mindseye.net.activation.LinearActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.loss.SqLossLayer;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.reducers.SumInputsLayer;
import com.simiacryptus.mindseye.net.util.VerboseWrapper;
import com.simiacryptus.mindseye.test.Tester;

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
    for (int ii = 0; ii < 3; ii++) {
      final int i = ii + ii * 3;
      convolution.kernel.set(new int[] { 0, 2, i }, 0.333);
      convolution.kernel.set(new int[] { 1, 1, i }, 0.333);
      convolution.kernel.set(new int[] { 2, 0, i }, 0.333);
    }
    convolution.freeze();
    return convolution;
  }

  public NNLayer<?> blur_3x4() {
    final DAGNetwork net = new PipelineNetwork();
    for (int i = 0; i < 3; i++) {
      net.add(blur_3());
    }
    return net;
  }

  public int[] outsize(final int[] inputSize, final int[] kernelSize) {
    return new int[] { inputSize[0] - kernelSize[0] + 1, inputSize[1] - kernelSize[1] + 1, inputSize[2] - kernelSize[2] + 1 };
  }

  @Test
  @Ignore
  public void testConvolution() throws Exception {

    final Tensor inputImage = DeconvolutionTest.toNDArray3(DeconvolutionTest.scale(ImageIO.read(getClass().getResourceAsStream("/monkey1.jpg")), .5));
    // final Tensor inputImage = Util.toNDArray1(render(new int[] { 200, 300 },
    // "Hello World"));

    final NNLayer<?> convolution = blur_3x4();
    // final NNLayer<?> convolution = blur_3();

    final int[] inputSize = inputImage.getDims();
    final int[] outSize = convolution.eval(new Tensor(inputSize)).data[0].getDims();
    final List<LabeledObject<Tensor>> data = new ArrayList<>();
    data.add(new LabeledObject<Tensor>(inputImage, "Ideal Input"));

    final DAGNetwork forwardConvolutionNet = new PipelineNetwork();
    forwardConvolutionNet.add(convolution);

    Util.report(data.stream().map(obj -> {
      final Tensor[] input = { obj.data };
      final NNResult output = forwardConvolutionNet.eval(input);

        return DeconvolutionTest.imageHtml(obj.data.toRgbImage(), new Tensor(outSize, output.data[0].getData()).toRgbImage());
    }));

  }

  @SuppressWarnings({ "unused", "serial" })
  @Test
  @Ignore
  public void testDeconvolution() throws Exception {

    // List<LabeledObject<Tensor>> data =
    // TestMNISTDev.trainingDataStream().limit(10).collect(Collectors.toList());
    BufferedImage read = ImageIO.read(getClass().getResourceAsStream("/monkey1.jpg"));
    final Tensor inputImage = DeconvolutionTest.toNDArray3(DeconvolutionTest.scale(read, .5));
    // final Tensor inputImage = Util.toNDArray1(render(new int[] { 200, 300 },
    // "Hello World"));
    // Tensor inputImage = Util.toNDArray3(render(new int[]{200,300}, "Hello
    // World"));

    final NNLayer<?> convolution = blur_3x4();
    // final NNLayer<?> convolution = blur_3();

    final int[] inputSize = inputImage.getDims();
    final int[] outSize = convolution.eval(new Tensor(inputSize)).data[0].getDims();
    final List<LabeledObject<Tensor>> data = new ArrayList<>();
    data.add(new LabeledObject<Tensor>(inputImage, "Ideal Input"));

    Util.report(data.stream().map(obj -> {

      final NNResult blurredImage = convolution.eval(new Tensor[] { obj.data });

      final Tensor zeroInput = new Tensor(inputSize);
      final PipelineNetwork net1 = new PipelineNetwork();
      BiasLayer bias = new BiasLayer(inputSize) {

        @Override
        public double[] add(final double[] input) {
          final double[] array = new double[input.length];
          for (int i = 0; i < array.length; i++) {
            final double v = input[i] + this.bias[i];
            array[i] = v < 0 ? 0 : v;
          }
          return array;
        }

      }.addWeights(() -> Util.R.get().nextGaussian() * 1e-5);
      net1.add(bias);
      final DAGNode modeledImageNode = net1.getHead();
      net1.add(convolution);

      final PipelineNetwork net2 = new PipelineNetwork(2);
      net2.add(new SupervisedNetwork(net1, new SqLossLayer()), net2.getInput(0), net2.getInput(1));
      final DAGNode imageRMS = net2.add(new VerboseWrapper("rms", new BiasLayer().freeze()));

      DAGNode image_entropy;
      {
        net2.add(new com.simiacryptus.mindseye.net.activation.AbsActivationLayer(), modeledImageNode);
        net2.add(new com.simiacryptus.mindseye.net.activation.L1NormalizationLayer());
        net2.add(new com.simiacryptus.mindseye.net.media.EntropyLayer());
        net2.add(new SumInputsLayer());
        // dagNetwork.add(new LinearActivationLayer().setWeights(new
        // double[]{-1.}));

        // Add 1 to output so product stays above 0 since this fitness function
        // is secondary
        // dagNetwork.add(new BiasLayer(new
        // int[]{1}).setWeights(i->1).freeze());
        image_entropy = net2.add(new VerboseWrapper("entropy", new BiasLayer().freeze()));
      }

      DAGNode edge_entropy_horizontal;
      {
        final ConvolutionSynapseLayer edgeFilter = new ConvolutionSynapseLayer(new int[] { 1, 2 }, 9);
        for (int ii = 0; ii < 3; ii++) {
          final int i = ii + ii * 3;
          edgeFilter.kernel.set(new int[] { 0, 0, i }, -1);
          edgeFilter.kernel.set(new int[] { 0, 1, i }, 1);
        }
        edgeFilter.freeze();
        net2.add(edgeFilter, modeledImageNode);

        net2.add(new com.simiacryptus.mindseye.net.activation.AbsActivationLayer());
        net2.add(new com.simiacryptus.mindseye.net.activation.L1NormalizationLayer());
        net2.add(new com.simiacryptus.mindseye.net.media.EntropyLayer());
        net2.add(new SumInputsLayer());

        // Add 1 to output so product stays above 0 since this fitness function
        // is secondary
        // dagNetwork.add(new BiasLayer(new
        // int[]{1}).setWeights(i->1).freeze());
        edge_entropy_horizontal = net2.add(new VerboseWrapper("edgeh", new BiasLayer().freeze()));
      }

      DAGNode edge_entropy_vertical;
      {
        final ConvolutionSynapseLayer edgeFilter = new ConvolutionSynapseLayer(new int[] { 2, 1 }, 9);
        for (int ii = 0; ii < 3; ii++) {
          final int i = ii + ii * 3;
          edgeFilter.kernel.set(new int[] { 0, 0, i }, -1);
          edgeFilter.kernel.set(new int[] { 1, 0, i }, 1);
        }
        edgeFilter.freeze();
        net2.add(edgeFilter, modeledImageNode);

        net2.add(new com.simiacryptus.mindseye.net.activation.AbsActivationLayer());
        net2.add(new com.simiacryptus.mindseye.net.activation.L1NormalizationLayer());
        net2.add(new com.simiacryptus.mindseye.net.media.EntropyLayer());
        net2.add(new SumInputsLayer());
        // dagNetwork.add(new LinearActivationLayer().setWeights(new
        // double[]{-1.}));

        // Add 1 to output so product stays above 0 since this fitness function
        // is secondary
        // dagNetwork.add(new BiasLayer(new
        // int[]{1}).setWeights(i->1).freeze());
        edge_entropy_vertical = net2.add(new VerboseWrapper("edgev", new BiasLayer().freeze()));
      }

      final LinearActivationLayer gate_rms = new LinearActivationLayer().setWeight(1).freeze();
      final LinearActivationLayer gate_entropy = new LinearActivationLayer().setWeight(1).freeze();
      final LinearActivationLayer gate_h = new LinearActivationLayer().setWeight(1).freeze();
      final LinearActivationLayer gate_v = new LinearActivationLayer().setWeight(1).freeze();

      final List<DAGNode> outs = new ArrayList<>();

      net2.add(gate_rms, imageRMS);
      outs.add(net2.getHead());
      net2.add(gate_entropy, image_entropy);
      outs.add(net2.getHead());
      // outs.add(dagNetwork.add(gate_h, edge_entropy_horizontal).getHead());
      // outs.add(dagNetwork.add(gate_v, edge_entropy_vertical).getHead());
      final VerboseWrapper combiner = new VerboseWrapper("product", new com.simiacryptus.mindseye.net.reducers.SumInputsLayer());
      net2.add(combiner, outs.stream().toArray(i -> new DAGNode[i]));
      final DAGNode combine = net2.getHead();

      final Tester trainer = new Tester() {
      }.setStaticRate(1.);

      // new NetInitializer().initialize(initPredictionNetwork);
      trainer.getGradientDescentTrainer().setNet(net2);
      trainer.getGradientDescentTrainer().setData(new Tensor[][] { { zeroInput, blurredImage.data[0] } });
      final TrainingContext trainingContext = new TrainingContext().setTimeout(1, java.util.concurrent.TimeUnit.MINUTES);
      try {
        trainer.setStaticRate(0.5).setMaxDynamicRate(1000000).setVerbose(true);

        // constrainedGDTrainer.setPrimaryNode(imageRMS);
        trainer.train(1., trainingContext);
        trainer.getDynamicRateTrainer().reset();

        // constrainedGDTrainer.setPrimaryNode(combine);
        // constrainedGDTrainer.setPrimaryNode(image_entropy);
        // constrainedGDTrainer.setPrimaryNode(edge_entropy_vertical);
        // constrainedGDTrainer.addConstraintNodes(imageRMS);
        trainer.train(-Double.MAX_VALUE, trainingContext);
        trainer.getDynamicRateTrainer().reset();

        // constrainedGDTrainer.setPrimaryNode(edge_entropy_vertical);
        // constrainedGDTrainer.addConstraintNodes(imageRMS);
        // trainer.train(1., trainingContext);
        // trainer.getDevtrainer().reset();

      } catch (final Exception e) {
        e.printStackTrace();
      }

      bias = (BiasLayer) trainer.getNet().getChild(bias.getId());
      final NNResult recovered = bias.eval(zeroInput);
      final PipelineNetwork verificationNet = new PipelineNetwork();
      verificationNet.add(bias);
      verificationNet.add(convolution);
      NNResult verification = verificationNet.eval(zeroInput);

        return DeconvolutionTest.imageHtml( //
                obj.data.toRgbImage(), //
                new Tensor(outSize, blurredImage.data[0].getData()).toRgbImage(), //
                new Tensor(inputSize, recovered.data[0].getData()).toRgbImage(), //
                new Tensor(outSize, verification.data[0].getData()).toRgbImage());
    }));

  }

  @Test
  @Ignore
  public void testDeconvolution2() throws Exception {

    // List<LabeledObject<Tensor>> data =
    // TestMNISTDev.trainingDataStream().limit(10).collect(Collectors.toList());
    final Tensor inputImage = DeconvolutionTest.toNDArray3(DeconvolutionTest.scale(ImageIO.read(getClass().getResourceAsStream("/monkey1.jpg")), .5));
    // final Tensor inputImage = Util.toNDArray1(render(new int[] { 200, 200 },
    // "Hello World"));
    // Tensor inputImage = TestMNISTDev.toNDArray3(render(new int[]{300,300},
    // "Hello World"));

    final NNLayer<?> convolution = blur_3x4();

    final int[] inputSize = inputImage.getDims();
    final int[] outSize = convolution.eval(new Tensor(inputSize)).data[0].getDims();
    final List<LabeledObject<Tensor>> data = new ArrayList<>();
    data.add(new LabeledObject<Tensor>(inputImage, "Ideal Input"));

    final DAGNetwork forwardConvolutionNet = new PipelineNetwork();
    forwardConvolutionNet.add(convolution);

    Util.report(data.stream().map(obj -> {
      final Tensor[] input = { obj.data };
      final NNResult output = forwardConvolutionNet.eval(input);
      final Tensor zeroInput = new Tensor(inputSize);
      BiasLayer bias = new BiasLayer(inputSize);
      final Tester trainer = new Tester().setStaticRate(1.);

      PipelineNetwork net1 = new PipelineNetwork();
      net1.add(bias);
      net1.add(convolution);
      trainer.init(new Tensor[][] { { zeroInput, output.data[0] } }, net1, new SqLossLayer());

      // trainer.add(new SupervisedTrainingParameters(
      // new PipelineNetwork().add(bias),
      // new Tensor[][] { { zeroInput, zeroInput } })
      // {
      // @Override
      // public Tensor getIdeal(NNResult eval, Tensor preset) {
      // Tensor retVal = preset.copy();
      // for (int i = 0; i < retVal.dim(); i++) {
      // double x = eval.data.getTrainingData()[i];
      // retVal.getTrainingData()[i] = ((x > -0.1)?x:0)*0.99;
      // }
      // return retVal;
      // }
      // }.setWeight(1));

      // trainer.add(new SupervisedTrainingParameters(
      // new PipelineNetwork().add(bias).add(new
      // com.simiacryptus.mindseye.layers.MaxEntLayer()),
      // new Tensor[][] { { zeroInput, new Tensor(1) } }).setWeight(0.1));

      // trainer.add(new SupervisedTrainingParameters(
      // new PipelineNetwork().add(bias).add(new
      // com.simiacryptus.mindseye.layers.MaxEntLayer().setFactor(1).setReverse(true)),
      // new Tensor[][] { { zeroInput, new Tensor(1) } }).setWeight(-0.1));

      final TrainingContext trainingContext = new TrainingContext().setTimeout(15, java.util.concurrent.TimeUnit.MINUTES);
      try {
        trainer.setStaticRate(0.5).setMaxDynamicRate(1000000).setVerbose(true).train(0.0, trainingContext);
        trainer.getDynamicRateTrainer().reset();
      } catch (final Exception e) {
        e.printStackTrace();
      }

      bias = (BiasLayer) trainer.getNet().getChild(bias.getId());
      final NNResult recovered = bias.eval(zeroInput);
      PipelineNetwork result;
      synchronized (new PipelineNetwork().add(bias)) {
        PipelineNetwork net2 = new PipelineNetwork();
        net2.add(bias);
        net2.add(convolution);
        result = new PipelineNetwork();
      }
      final NNResult tested = result.eval(zeroInput);

        return DeconvolutionTest.imageHtml(obj.data.toRgbImage(), new Tensor(outSize, output.data[0].getData()).toRgbImage(), new Tensor(inputSize, recovered.data[0].getData()).toRgbImage(),
                new Tensor(outSize, tested.data[0].getData()).toRgbImage());
    }));

  }

  public static String imageHtml(final BufferedImage... imgArray) {
    return Stream.of(imgArray).map(img -> Util.toInlineImage(img, "")).reduce((a, b) -> a + b).get();
  }

  public static Tensor toNDArray3(final BufferedImage img) {
    final Tensor a = new Tensor(img.getWidth(), img.getHeight(), 3);
    for (int x = 0; x < img.getWidth(); x++) {
      for (int y = 0; y < img.getHeight(); y++) {
        a.set(new int[] { x, y, 0 }, img.getRGB(x, y) & 0xFF);
        a.set(new int[] { x, y, 1 }, img.getRGB(x, y) >> 8 & 0xFF);
        a.set(new int[] { x, y, 2 }, img.getRGB(x, y) >> 16 & 0x0FF);
      }
    }
    return a;
  }

}
