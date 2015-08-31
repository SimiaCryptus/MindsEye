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
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.PipelineNetwork;
import com.simiacryptus.mindseye.training.Trainer;
import com.simiacryptus.mindseye.util.Util;

public abstract class ClassificationTestBase {
  
  static final Logger log = LoggerFactory.getLogger(ClassificationTestBase.class);

  public abstract PipelineNetwork buildNetwork();

  public ClassificationTestBase() {
    super();
  }

  public Trainer buildTrainer(final NDArray[][] samples, final PipelineNetwork net) {
    return net.trainer(samples);
  }

  public NDArray[][] getTrainingData(final int dimensions, final List<Function<Void, double[]>> populations) throws FileNotFoundException, IOException {
    return getTrainingData(dimensions, populations, 100);
  }

  public NDArray[][] getTrainingData(final int dimensions, final List<Function<Void, double[]>> populations, final int sampleN) throws FileNotFoundException, IOException {
    final int[] inputSize = new int[] { dimensions };
    final int[] outSize = new int[] { populations.size() };
    final NDArray[][] samples = IntStream.range(0, populations.size()).mapToObj(x -> x)
        .flatMap(p -> IntStream.range(0, sampleN).mapToObj(i -> {
          return new NDArray[] {
              new NDArray(inputSize, populations.get(p).apply(null)),
              new NDArray(inputSize, IntStream.range(0, outSize[0]).mapToDouble(x -> p.equals(x) ? 1 : 0).toArray()) };
        })).toArray(i -> new NDArray[i][]);
    return samples;
  }

  public static class ClassificationResultMetrics {
    public double pts = 0;
    public double sumSqErr;
    public double classificationAccuracy;
    public NDArray classificationMatrix;

    public ClassificationResultMetrics(int categories) {
      this.classificationMatrix = new NDArray(categories,categories);
    }

    @Override
    public String toString() {
      StringBuilder builder = new StringBuilder();
      builder.append("ClassificationResultMetrics [");
      if(pts>0){
        builder.append("error=");
        builder.append(Math.sqrt(sumSqErr/pts));
      }
      builder.append(", accuracy=");
      builder.append(classificationAccuracy);
      builder.append(", classificationMatrix=");
      builder.append(classificationMatrix);
      builder.append("]");
      return builder.toString();
    }

    
    
  }
  
  boolean drawBG = true;
  public void test(final NDArray[][] samples) throws FileNotFoundException, IOException {
    final PipelineNetwork net = buildNetwork();
    final Trainer trainer = buildTrainer(samples, net);
    final Map<BufferedImage,ClassificationResultMetrics> images = new HashMap<>();
    int categories = samples[0][1].dim();
    trainer.handler.add(n -> {
      try {
        ClassificationResultMetrics correct = new ClassificationResultMetrics(categories);
        final BufferedImage img = new BufferedImage(500, 500, BufferedImage.TYPE_INT_RGB) {
          {
            if(drawBG) {
              for (int xpx = 0; xpx < getWidth(); xpx++) {
                for (int ypx = 0; ypx < getHeight(); ypx++) {
                  final double xf = (xpx * 1. / getWidth() - .5) * 6;
                  final double yf = (ypx * 1. / getHeight() - .5) * 6;
                  final NNResult eval = n.getNetwork().get(0).eval(new NDArray(new int[] { 2 }, new double[] { xf, yf }));
                  int classificationActual = outputToClassification(eval.data);
                  int color = 0 == classificationActual ? 0x1F0000 : 0x001F00;
                  this.setRGB(xpx, ypx, color);
                }
              }
            }
            final Graphics2D g = (Graphics2D) getGraphics();
            correct.pts++;
            correct.classificationAccuracy = (Stream.of(samples).mapToDouble(pt -> {
              final NDArray expectedOutput = pt[1];
              NDArray input = pt[0];
              final NDArray actualOutput = n.getNetwork().get(0).eval(input).data;
              correct.sumSqErr += IntStream.range(0, actualOutput.dim()).mapToDouble(i -> {
                double x = expectedOutput.get(i)-actualOutput.get(i);
                return x*x;
              }).average().getAsDouble();
              
              final int classificationExpected = outputToClassification(expectedOutput);
              final int classificationActual = outputToClassification(actualOutput);
              double[] coords = inputToXY(input, classificationActual, classificationExpected);
              final double xf = coords[0];
              final double yf = coords[1];
              final int xpx = (int) ((xf + 3) / 6 * getHeight());
              final int ypx = (int) ((yf + 3) / 6 * getHeight());
              Color color = getColor(input, classificationActual, classificationExpected);
              g.setColor(color);
              g.drawOval(xpx - 1, ypx - 1, 2, 2);
              correct.classificationMatrix.add(new int[]{classificationExpected,classificationActual}, 1.);
              return classificationExpected==classificationActual?1.:0.;
            }).average().getAsDouble());
          }
        };
        log.debug(correct.toString());
        images.put(img, correct);
      } catch (final Exception e) {
        e.printStackTrace();
      }
      return null;
    });
    try {
      verify(trainer);
    } finally {
      Stream<String> map = images.entrySet().stream().map(e -> Util.toInlineImage(e.getKey(), e.getValue().toString()));
      String[] array = map.toArray(i -> new String[i]);
      Util.report(array);
    }
  }

  public double[] inputToXY(NDArray input, int classificationActual, int classificationExpected) {
    final double xf = input.get(0);
    final double yf = input.get(1);
    return new double[] { xf, yf };
  }

  public Integer outputToClassification(NDArray actual) {
    return IntStream.range(0, actual.dim()).mapToObj(o -> o).max(Comparator.comparing(o -> actual.get((int) o))).get();
  }
  public void verify(final Trainer trainer) {
    trainer.verifyConvergence(0, 0.0, 10);
  }

  List<Color> colorMap = Arrays.asList(Color.RED, Color.GREEN,randomColor(),randomColor(),randomColor(),randomColor(),randomColor(),randomColor(),randomColor(),randomColor());
  
  public Color getColor(NDArray input, int classificationActual, final int classificationExpected) {
    return colorMap.get(classificationExpected);
  }

  public Color randomColor() {
    return new Color(Util.R.get().nextInt(255),Util.R.get().nextInt(255),Util.R.get().nextInt(255));
  }
  
}