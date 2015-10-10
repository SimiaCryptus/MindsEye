package com.simiacryptus.mindseye.test.demo.mnist;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.junit.Test;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.LabeledObject;
import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.NNLayer;
import com.simiacryptus.mindseye.core.NNResult;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.test.Tester;
import com.simiacryptus.mindseye.test.demo.ClassificationTestBase;
import com.simiacryptus.mindseye.test.dev.MNIST;
import com.simiacryptus.mindseye.test.dev.SimpleMNIST;

public class MNISTClassificationTests extends ClassificationTestBase {

  private static final List<Color> colorMap = Arrays.asList(Color.WHITE, Color.RED, Color.ORANGE, Color.YELLOW, Color.GREEN, Color.BLUE, Color.decode("0xee82ee"), Color.PINK,
      Color.GRAY, ClassificationTestBase.randomColor(), ClassificationTestBase.randomColor());

  public MNISTClassificationTests() {
    super();
  }

  @Override
  public NNLayer<DAGNetwork> buildNetwork() {
    final int[] inputSize = new int[] { 28, 28, 1 };
    final int[] outSize = new int[] { 10 };
    DAGNetwork net = new DAGNetwork();
    net = net.add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize));
    net = net.add(new BiasLayer(outSize));
    // net = net.add(new MinMaxFilterLayer());
    net = net.add(new SoftmaxActivationLayer());
    return net;
  }

  @Override
  public Tester buildTrainer(final NDArray[][] samples, final NNLayer<DAGNetwork> net) {
    final EntropyLossLayer lossLayer = new EntropyLossLayer();
    final Tester trainer = new Tester().init(samples, net, lossLayer).setVerbose(true);
    trainer.setVerbose(true);
    trainer.trainingContext().setTimeout(5, java.util.concurrent.TimeUnit.MINUTES);
    return trainer;
  }

  @Override
  public BufferedImage draw(final NDArray[][] samples, final NNLayer<?> mainNetwork, final ClassificationResultMetrics correct) {
    final BufferedImage img = new BufferedImage(width(), height(), BufferedImage.TYPE_INT_RGB) {
      {
        final Graphics2D g = (Graphics2D) getGraphics();
        correct.pts++;
        correct.classificationAccuracy = Stream.of(samples).mapToDouble(pt -> {
          final NDArray expectedOutput = pt[1];
          final NDArray[] array = pt;
          final NNResult output = mainNetwork.eval(array);
          final NDArray actualOutput = output.data;
          correct.sumSqErr += IntStream.range(0, actualOutput.dim()).mapToDouble(i -> {
            final double x = expectedOutput.get(i) - actualOutput.get(i);
            return x * x;
          }).average().getAsDouble();

          final int classificationExpected = outputToClassification(expectedOutput);
          final int classificationActual = outputToClassification(actualOutput);
          final double n = numberOfSymbols();
          final double[] c = new double[] { //
              (classificationActual + Util.R.get().nextDouble()) / (n + 1), //
              (classificationExpected + Util.R.get().nextDouble()) / (n + 1) //
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

  public boolean filter(final LabeledObject<NDArray> item) {
    if (item.label.equals("[0]"))
      return true;
    if (item.label.equals("[5]"))
      return true;
    if (item.label.equals("[9]"))
      return true;
    return true;
  }

  @Override
  public List<Color> getColorMap() {
    return colorMap;
  }

  public double numberOfSymbols() {
    return 10.;
  }

  private String remap(final String label) {
    switch (label) {
    // case "[0]":
    // return "[5]";
    // case "[5]":
    // return "[9]";
    // case "[9]":
    // return "[0]";
    default:
      return label;
    }
  }

  @Test
  public void test() throws Exception {
    final int hash = Util.R.get().nextInt();
    log.debug(String.format("Shuffle hash: 0x%s", Integer.toHexString(hash)));
    final NDArray[][] data = transformTrainingData(hash, MNIST.trainingDataStream());
    final NDArray[][] trainingData = java.util.Arrays.copyOfRange(data, 0, data.length);
    final NDArray[][] validationData = transformTrainingData(hash, MNIST.validationDataStream());
    test(trainingData, validationData);
  }

  public NDArray[][] transformTrainingData(final int hash, final Stream<LabeledObject<NDArray>> mnistStream) {
    final NDArray[][] data = mnistStream.filter(this::filter).collect(java.util.stream.Collectors.toList()).stream()
        .sorted(java.util.Comparator.comparingInt(obj -> 0xEFFFFFFF & (System.identityHashCode(obj) ^ hash)))
        // .limit(1000)
        .collect(java.util.stream.Collectors.toList()).parallelStream()
        // .limit(1000)
        .map(obj -> new LabeledObject<>(obj.data.reformat(28, 28, 1), obj.label)).map(obj -> {
          final int out = SimpleMNIST.toOut(remap(obj.label));
          final NDArray output = SimpleMNIST.toOutNDArray(out, 10);
          return new NDArray[] { obj.data, output };
        }).toArray(i -> new NDArray[i][]);
    return data;
  }

  @Override
  public void verify(final Tester trainer) {
    trainer.verifyConvergence(0.00001, 1);
  }

}
