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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.EntropyLossLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.test.Tester;
import com.simiacryptus.mindseye.util.Util;

public abstract class ClassificationTestBase {

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

  static final Logger log = LoggerFactory.getLogger(ClassificationTestBase.class);

  public static Color randomColor() {
    final Random r = Util.R.get();
    return new Color(r.nextInt(255), r.nextInt(255), r.nextInt(255));
  }

  public ClassificationTestBase() {
    super();
  }

  public abstract NNLayer<DAGNetwork> buildNetwork();

  public Tester buildTrainer(final NDArray[][] samples, final NNLayer<DAGNetwork> net) {
    return new Tester().init(samples, net, (NNLayer<?>) new EntropyLossLayer());
  }

  public NDArray[][] getTrainingData(final int dimensions, final List<Function<Void, double[]>> populations) throws FileNotFoundException, IOException {
    return getTrainingData(dimensions, populations, 100);
  }

  public NDArray[][] getTrainingData(final int dimensions, final List<Function<Void, double[]>> populations, final int sampleN) throws FileNotFoundException, IOException {
    final int[] inputSize = new int[] { dimensions };
    final int[] outSize = new int[] { populations.size() };
    final NDArray[][] samples = IntStream.range(0, populations.size()).mapToObj(x -> x).flatMap(p -> IntStream.range(0, getSampleSize(p,sampleN)).mapToObj(i -> {
      return new NDArray[] { new NDArray(inputSize, populations.get(p).apply(null)),
          new NDArray(inputSize, IntStream.range(0, outSize[0]).mapToDouble(x -> p.equals(x) ? 1 : 0).toArray()) };
    })).toArray(i -> new NDArray[i][]);
    return samples;
  }

  protected int getSampleSize(Integer populationIndex, int defaultNum) {
    return defaultNum;
  }

  public Integer outputToClassification(final NDArray actual) {
    return IntStream.range(0, actual.dim()).mapToObj(o -> o).max(Comparator.comparing(o -> actual.get((int) o))).get();
  }

  public void test(final NDArray[][] samples) throws FileNotFoundException, IOException {
    test(samples,samples);
  }

  public void test(final NDArray[][] trainingsamples,final NDArray[][] validationsamples) throws FileNotFoundException, IOException {
    final NNLayer<DAGNetwork> net = buildNetwork();
    final Tester trainer = buildTrainer(trainingsamples, net);
    final Map<BufferedImage, String> images = new HashMap<>();
    final int categories = trainingsamples[0][1].dim();
    trainer.handler.add((n, trainingContext) -> {
      try {
        NNLayer<?> mainNetwork = n.getChild(net.id);
        final ClassificationResultMetrics correct = new ClassificationResultMetrics(categories);
        final BufferedImage img = draw(validationsamples, mainNetwork, correct);
        final String label = correct.toString() + " \n" + trainingContext.toString();
        ClassificationTestBase.log.debug(label);
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

  public BufferedImage draw(final NDArray[][] samples, NNLayer<?> mainNetwork, final ClassificationResultMetrics correct) {
    final BufferedImage img = new BufferedImage(width(), height(), BufferedImage.TYPE_INT_RGB) {
      {
        for (int xpx = 0; xpx < getWidth(); xpx++) {
          for (int ypx = 0; ypx < getHeight(); ypx++) {
            final double xf = (xpx * 1. / getWidth() - .5) * 6;
            final double yf = (ypx * 1. / getHeight() - .5) * 6;
            final NNResult eval = mainNetwork.eval(new NDArray(new int[] { 2 }, new double[] { xf, yf }));
            final int classificationActual = outputToClassification(eval.data);
            final int color = 0 == classificationActual ? 0x1F0000 : 0x001F00;
            this.setRGB(xpx, ypx, color);
          }
        }

        final Graphics2D g = (Graphics2D) getGraphics();
        correct.pts++;
        correct.classificationAccuracy = Stream.of(samples).mapToDouble(pt -> {
          final NDArray expectedOutput = pt[1];
          final NDArray input = pt[0];
          final NDArray[] array = pt;
          final NNResult output = mainNetwork.eval(array);
          final NDArray actualOutput = output.data;
          correct.sumSqErr += IntStream.range(0, actualOutput.dim()).mapToDouble(i -> {
            final double x = expectedOutput.get(i) - actualOutput.get(i);
            return x * x;
          }).average().getAsDouble();

          final int classificationExpected = outputToClassification(expectedOutput);
          final int classificationActual = outputToClassification(actualOutput);
          final int xpx = (int) ((input.get(0) + 3) / 6 * getHeight());
          final int ypx = (int) ((input.get(1) + 3) / 6 * getHeight());
          final Color color = getColorMap().get(classificationExpected);
          g.setColor(color);
          g.drawOval(xpx, ypx, 1, 1);
          correct.classificationMatrix.add(new int[] { classificationExpected, classificationActual }, 1.);
          return classificationExpected == classificationActual ? 1. : 0.;
        }).average().getAsDouble();
      }
    };
    return img;
  }

  public int height() {
    return 500;
  }

  public int width() {
    return 500;
  }

  public void verify(final Tester trainer) {
    trainer.verifyConvergence(0.0, 10);
  }


  private static final List<Color> colorMap = Arrays.asList(Color.RED, Color.GREEN, ClassificationTestBase.randomColor(), ClassificationTestBase.randomColor(),
      ClassificationTestBase.randomColor(), ClassificationTestBase.randomColor(), ClassificationTestBase.randomColor(), ClassificationTestBase.randomColor(),
      ClassificationTestBase.randomColor(), ClassificationTestBase.randomColor());
  public List<Color> getColorMap() {
    return colorMap;
  }

}
