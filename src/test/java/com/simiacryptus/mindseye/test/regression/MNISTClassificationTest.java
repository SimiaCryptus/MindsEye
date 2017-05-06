package com.simiacryptus.mindseye.test.regression;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.util.ml.Tensor;
import org.junit.Ignore;
import org.junit.Test;

import com.simiacryptus.util.Util;
import com.simiacryptus.util.test.LabeledObject;
import com.simiacryptus.mindseye.training.TrainingContext;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;
import com.simiacryptus.mindseye.net.PipelineNetwork;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.dev.DenseSynapseLayerJBLAS;
import com.simiacryptus.mindseye.net.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.test.Tester;
import com.simiacryptus.mindseye.test.demo.ClassificationTestBase;
import com.simiacryptus.util.test.MNIST;

public class MNISTClassificationTest extends ClassificationTestBase {

  private static final List<Color> colorMap = Arrays.asList(Color.WHITE, Color.RED, Color.ORANGE, Color.YELLOW, Color.GREEN, Color.BLUE, Color.decode("0xee82ee"), Color.PINK,
      Color.GRAY, ClassificationTestBase.randomColor(), ClassificationTestBase.randomColor());

  public MNISTClassificationTest() {
    super();
  }

  @Override
  public NNLayer<DAGNetwork> buildNetwork() {
    final int[] inputSize = new int[] { 28, 28, 1 };
    final int[] outSize = new int[] { 10 };
    DAGNetwork net = new PipelineNetwork();
    net.add(new DenseSynapseLayerJBLAS(inputSize, outSize).setWeights(()->Util.R.get().nextGaussian()*0.));
    net.add(new BiasLayer(outSize));
    // net = net.add(new MinMaxFilterLayer());
    net.add(new SoftmaxActivationLayer());
    return net;
  }

  @Override
  public Tester buildTrainer(final Tensor[][] samples, final NNLayer<DAGNetwork> net) {
    final EntropyLossLayer lossLayer = new EntropyLossLayer();
    final Tester trainer = new Tester().init(samples, net, lossLayer).setVerbose(true);
    trainer.setVerbose(true);
    trainer.trainingContext().setTimeout(5, java.util.concurrent.TimeUnit.MINUTES);
    return trainer;
  }

  @Override
  public BufferedImage draw(final Tensor[][] samples, final NNLayer<?> mainNetwork, final ClassificationResultMetrics correct) {
    final BufferedImage img = new BufferedImage(width(), height(), BufferedImage.TYPE_INT_RGB) {
      {
        final Graphics2D g = (Graphics2D) getGraphics();
        correct.pts++;
        correct.classificationAccuracy = Stream.of(samples).mapToDouble(pt -> {
          final Tensor expectedOutput = pt[1];
          final Tensor[] array = pt;
          final NNResult output = mainNetwork.eval(array);
          final Tensor actualOutput = output.data[0];
          correct.sumSqErr += IntStream.range(0, actualOutput.dim()).mapToDouble(i -> {
            final double x = expectedOutput.get(i) - actualOutput.get(i);
            return x * x;
          }).average().getAsDouble();

          final int classificationExpected = outputToClassification(expectedOutput);
          final int classificationActual = outputToClassification(actualOutput);
          final double n = numberOfSymbols();
          final double[] c = new double[] { //
              (classificationActual + Util.R.get().nextDouble()) / (n), //
              (classificationExpected + Util.R.get().nextDouble()) / (n) //
          };
          final double[] coords = new double[] { c[0] * 6 - 3, c[1] * 6 - 3 };
          final double xf = coords[0];
          final double yf = coords[1];
          final int xpx = (int) ((xf + 3) / 6 * getHeight());
          final int ypx = (int) ((yf + 3) / 6 * getHeight());
          final Color color = getColorMap().get(Util.R.get().nextBoolean() ? classificationActual : classificationExpected);
          g.setColor(color);
          g.drawOval(xpx, ypx, 1, 1);
          correct.classificationMatrix.add(new int[] { classificationExpected, classificationActual }, 1.);
          return classificationExpected == classificationActual ? 1. : 0.;
        }).average().getAsDouble();
      }
    };
    return img;
  }

  @Override
  public List<Color> getColorMap() {
    return colorMap;
  }

  public double numberOfSymbols() {
    return 10.;
  }

  @Test
  @Ignore
  public void test() throws Exception {
    final int hash = Util.R.get().nextInt();
    log.debug(String.format("Shuffle hash: 0x%s", Integer.toHexString(hash)));
    final Tensor[][] trainingData = transformTrainingData(hash, MNIST.trainingDataStream());
    final Tensor[][] validationData = transformTrainingData(hash, MNIST.validationDataStream());
    test(trainingData, validationData);
  }

  public Tensor[][] transformTrainingData(final int hash, final Stream<LabeledObject<Tensor>> mnistStream) {
    final Tensor[][] data = mnistStream
        //.collect(java.util.stream.Collectors.toList()).stream()
        .collect(java.util.stream.Collectors.toList()).parallelStream()
        .sorted(java.util.Comparator.comparingInt(obj -> 0xEFFFFFFF & (System.identityHashCode(obj) ^ hash)))
        .map(obj -> new LabeledObject<>(obj.data.reformat(28, 28, 1), obj.label)).map(obj -> {
          final int out = MNISTClassificationTest.toOut(obj.label);
          final Tensor output = MNISTClassificationTest.toOutNDArray(out, 10);
          return new Tensor[] { obj.data, output };
        }).toArray(i -> new Tensor[i][]);
    return data;
  }

  @Override
  public int height() {
    return (int) (.5*super.height());
  }

  @Override
  public int width() {
    return (int) (.5*super.width());
  }

  public static int toOut(final String label) {
    for (int i = 0; i < 10; i++) {
      if (label.equals("[" + i + "]"))
        return i;
    }
    throw new RuntimeException();
  }

  public static Tensor toOutNDArray(final int out, final int max) {
    final Tensor tensor = new Tensor(max);
    tensor.set(out, 1);
    return tensor;
  }

  @Override
  public void train(final NNLayer<DAGNetwork> net, final Tensor[][] trainingsamples, final BiFunction<DAGNetwork, TrainingContext, Void> resultHandler) {
    final Tester trainer = buildTrainer(trainingsamples, net);
    trainer.handler.add(resultHandler);
    trainer.verifyConvergence(0.35, 1);  
  }

}
