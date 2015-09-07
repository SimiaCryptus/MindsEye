package com.simiacryptus.mindseye.test.demo;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.data.GaussianDistribution;
import com.simiacryptus.mindseye.data.Simple2DCircle;
import com.simiacryptus.mindseye.data.Simple2DLine;
import com.simiacryptus.mindseye.data.SnakeDistribution;
import com.simiacryptus.mindseye.data.UnionDistribution;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.DAGNetwork;
import com.simiacryptus.mindseye.training.Tester;
import com.simiacryptus.mindseye.util.Util;

public class EncogClassificationTests {

  public static class ClassificationResultMetrics {
    public double classificationAccuracy;
    public NDArray classificationMatrix;
    public double pts = 0;
    public double sumSqErr;

    public ClassificationResultMetrics(final int categories) {
      this.classificationMatrix = new NDArray(categories, categories);
    }

    @Override
    public String toString() {
      final StringBuilder builder = new StringBuilder();
      builder.append("ClassificationResultMetrics [");
      if (this.pts > 0) {
        builder.append("error=");
        builder.append(Math.sqrt(this.sumSqErr / this.pts));
      }
      builder.append(", accuracy=");
      builder.append(this.classificationAccuracy);
      builder.append(", classificationMatrix=");
      builder.append(this.classificationMatrix);
      builder.append("]");
      return builder.toString();
    }

  }

  static final List<Color> colorMap = Arrays.asList(Color.RED, Color.GREEN, EncogClassificationTests.randomColor(), EncogClassificationTests.randomColor(),
      EncogClassificationTests.randomColor(), EncogClassificationTests.randomColor(), EncogClassificationTests.randomColor(), EncogClassificationTests.randomColor(),
      EncogClassificationTests.randomColor(), EncogClassificationTests.randomColor());

  static final Logger log = LoggerFactory.getLogger(EncogClassificationTests.class);

  public static Color randomColor() {
    final Random r = Util.R.get();
    return new Color(r.nextInt(255), r.nextInt(255), r.nextInt(255));
  }

  boolean drawBG = true;

  public EncogClassificationTests() {
    super();
  }

  public Tester buildTrainer(final NDArray[][] samples, final DAGNetwork net) {
    return net.trainer(samples);
  }

  public Color getColor(final NDArray input, final int classificationActual, final int classificationExpected) {
    final Color color = EncogClassificationTests.colorMap.get(classificationExpected);
    return color;
  }

  public NDArray[][] getTrainingData(final int dimensions, final List<Function<Void, double[]>> populations) throws FileNotFoundException, IOException {
    return getTrainingData(dimensions, populations, 100);
  }

  public NDArray[][] getTrainingData(final int dimensions, final List<Function<Void, double[]>> populations, final int sampleN) throws FileNotFoundException, IOException {
    final int[] inputSize = new int[] { dimensions };
    final int[] outSize = new int[] { populations.size() };
    final NDArray[][] samples = IntStream.range(0, populations.size()).mapToObj(x -> x).flatMap(p -> IntStream.range(0, sampleN).mapToObj(i -> {
      return new NDArray[] { new NDArray(inputSize, populations.get(p).apply(null)),
          new NDArray(inputSize, IntStream.range(0, outSize[0]).mapToDouble(x -> p.equals(x) ? 1 : 0).toArray()) };
    })).toArray(i -> new NDArray[i][]);
    return samples;
  }

  public double[] inputToXY(final NDArray input, final int classificationActual, final int classificationExpected) {
    final double xf = input.get(0);
    final double yf = input.get(1);
    return new double[] { xf, yf };
  }

  public Integer outputToClassification(final NDArray actual) {
    return IntStream.range(0, actual.dim()).mapToObj(o -> o).max(Comparator.comparing(o -> actual.get((int) o))).get();
  }

  public void test(final NDArray[][] samples) throws FileNotFoundException, IOException {
    final DAGNetwork net = null;
    final Tester trainer = buildTrainer(samples, net);
    final Map<BufferedImage, String> images = new HashMap<>();
    final int categories = samples[0][1].dim();
    trainer.handler.add((n, trainingContext) -> {
      try {
        final ClassificationResultMetrics correct = new ClassificationResultMetrics(categories);
        final BufferedImage img = new BufferedImage(500, 500, BufferedImage.TYPE_INT_RGB) {
          {
            if (EncogClassificationTests.this.drawBG) {
              for (int xpx = 0; xpx < getWidth(); xpx++) {
                for (int ypx = 0; ypx < getHeight(); ypx++) {
                  final double xf = (xpx * 1. / getWidth() - .5) * 6;
                  final double yf = (ypx * 1. / getHeight() - .5) * 6;
                  final NNResult eval = trainer.getInner().getNet().eval(new NDArray(new int[] { 2 }, new double[] { xf, yf }));
                  final int classificationActual = outputToClassification(eval.data);
                  final int color = 0 == classificationActual ? 0x1F0000 : 0x001F00;
                  this.setRGB(xpx, ypx, color);
                }
              }
            }
            final Graphics2D g = (Graphics2D) getGraphics();
            correct.pts++;
            correct.classificationAccuracy = Stream.of(samples).mapToDouble(pt -> {
              final NDArray expectedOutput = pt[1];
              final NDArray input = pt[0];
              final NNResult output = trainer.getInner().getNet().eval(input);
              final NDArray actualOutput = output.data;
              correct.sumSqErr += IntStream.range(0, actualOutput.dim()).mapToDouble(i -> {
                final double x = expectedOutput.get(i) - actualOutput.get(i);
                return x * x;
              }).average().getAsDouble();

              final int classificationExpected = outputToClassification(expectedOutput);
              final int classificationActual = outputToClassification(actualOutput);
              final double[] coords = inputToXY(input, classificationActual, classificationExpected);
              final double xf = coords[0];
              final double yf = coords[1];
              final int xpx = (int) ((xf + 3) / 6 * getHeight());
              final int ypx = (int) ((yf + 3) / 6 * getHeight());
              final Color color = getColor(input, classificationActual, classificationExpected);
              g.setColor(color);
              g.drawOval(xpx, ypx, 1, 1);
              correct.classificationMatrix.add(new int[] { classificationExpected, classificationActual }, 1.);
              return classificationExpected == classificationActual ? 1. : 0.;
            }).average().getAsDouble();
          }
        };
        final String label = correct.toString() + " \n" + trainingContext.toString();
        EncogClassificationTests.log.debug(label);
        images.put(img, label);
      } catch (final Exception e) {
        e.printStackTrace();
      }
      return null;
    });
    try {
      verify(trainer);
    } finally {
      final Stream<String> map = images.entrySet().stream().map(e -> Util.toInlineImage(e.getKey(), e.getValue().toString()));
      final String[] array = map.toArray(i -> new String[i]);
      Util.report(array);
    }
  }

  @Test(expected = RuntimeException.class)
  @Ignore
  public void test_Gaussians() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>>asList(new GaussianDistribution(2), new GaussianDistribution(2)), 100));
  }

  @Test(expected = RuntimeException.class)
  public void test_II() throws Exception {
    final double e = 1e-1;
    test(getTrainingData(2, Arrays.<Function<Void, double[]>>asList(new Simple2DLine(new double[] { -1, -1 }, new double[] { 1, 1 }),
        new Simple2DLine(new double[] { -1 + e, -1 - e }, new double[] { 1 + e, 1 - e })), 100));
  }

  @Test(expected = RuntimeException.class)
  public void test_III() throws Exception {
    final double e = 1e-1;
    test(getTrainingData(2, Arrays.<Function<Void, double[]>>asList(new UnionDistribution(new Simple2DLine(new double[] { -1 + e, -1 - e }, new double[] { 1 + e, 1 - e }),
        new Simple2DLine(new double[] { -1 - e, -1 + e }, new double[] { 1 - e, 1 + e })), new Simple2DLine(new double[] { -1, -1 }, new double[] { 1, 1 })), 100));
  }

  @Test(expected = RuntimeException.class)
  @Ignore
  public void test_Lines() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>>asList(new Simple2DLine(Util.R.get()), new Simple2DLine(Util.R.get())), 100));
  }

  @Test(expected = RuntimeException.class)
  public void test_O() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>>asList(new UnionDistribution(new Simple2DCircle(2, new double[] { 0, 0 })),
        new UnionDistribution(new Simple2DCircle(0.1, new double[] { 0, 0 }))), 100));
  }

  @Test(expected = RuntimeException.class)
  public void test_O2() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>>asList(new UnionDistribution(new Simple2DCircle(2, new double[] { 0, 0 })),
        new UnionDistribution(new Simple2DCircle(1.75, new double[] { 0, 0 }))), 100));
  }

  @Test(expected = RuntimeException.class)
  public void test_O22() throws Exception {
    test(getTrainingData(2,
        Arrays.<Function<Void, double[]>>asList(
            new UnionDistribution(new Simple2DCircle(2, new double[] { 0, 0 }), new Simple2DCircle(2 * (1.75 * 1.75) / 4, new double[] { 0, 0 })),
            new UnionDistribution(new Simple2DCircle(1.75, new double[] { 0, 0 }))),
        100));
  }

  @Test(expected = RuntimeException.class)
  // @Ignore
  public void test_O3() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>>asList(new UnionDistribution(new GaussianDistribution(2, new double[] { 0, 0 }, 1)),
        new UnionDistribution(new Simple2DCircle(.5, new double[] { 0, 0 }))), 1000));
  }

  @Test(expected = RuntimeException.class)
  public void test_oo() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>>asList(new UnionDistribution(new Simple2DCircle(1, new double[] { -0.5, 0 })),
        new UnionDistribution(new Simple2DCircle(1, new double[] { 0.5, 0 }))), 100));
  }

  @Test(expected = RuntimeException.class)
  public void test_simple() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>>asList(new UnionDistribution(new GaussianDistribution(2, new double[] { 0, 0 }, 0.1)),
        new UnionDistribution(new GaussianDistribution(2, new double[] { 1, 1 }, 0.1))), 100));
  }

  @Test(expected = RuntimeException.class)
  @Ignore
  public void test_snakes() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>>asList(new SnakeDistribution(2, Util.R.get(), 7, 0.01), new SnakeDistribution(2, Util.R.get(), 7, 0.01)), 100));
  }

  @Test(expected = RuntimeException.class)
  public void test_sos() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>>asList(new UnionDistribution(new GaussianDistribution(2, new double[] { 0, 0 }, 0.1)),
        new UnionDistribution(new GaussianDistribution(2, new double[] { -1, 0 }, 0.1), new GaussianDistribution(2, new double[] { 1, 0 }, 0.1))), 100));
  }

  @Test(expected = RuntimeException.class)
  public void test_X() throws Exception {
    test(getTrainingData(2,
        Arrays.<Function<Void, double[]>>asList(new Simple2DLine(new double[] { -1, -1 }, new double[] { 1, 1 }), new Simple2DLine(new double[] { -1, 1 }, new double[] { 1, -1 })),
        100));
  }

  @Test(expected = RuntimeException.class)
  public void test_xor() throws Exception {
    test(getTrainingData(2,
        Arrays.<Function<Void, double[]>>asList(
            new UnionDistribution(new GaussianDistribution(2, new double[] { 0, 0 }, 0.1), new GaussianDistribution(2, new double[] { 1, 1 }, 0.1)),
            new UnionDistribution(new GaussianDistribution(2, new double[] { 1, 0 }, 0.1), new GaussianDistribution(2, new double[] { 0, 1 }, 0.1))),
        100));
  }

  public void verify(final Tester trainer) {
    trainer.verifyConvergence(0.0, 10);
  }

}
