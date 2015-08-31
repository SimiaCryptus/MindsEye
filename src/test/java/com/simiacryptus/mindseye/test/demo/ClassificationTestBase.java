package com.simiacryptus.mindseye.test.demo;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.PipelineNetwork;
import com.simiacryptus.mindseye.training.Trainer;
import com.simiacryptus.mindseye.util.Util;

public abstract class ClassificationTestBase {
  
  static final Logger log = LoggerFactory.getLogger(SoftmaxTests1.class);

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

  public void test(final NDArray[][] samples) throws FileNotFoundException, IOException {
    final PipelineNetwork net = buildNetwork();
    final Trainer trainer = buildTrainer(samples, net);
    final ArrayList<BufferedImage> images = new ArrayList<>();
    trainer.handler.add(n -> {
      try {
        final BufferedImage img = new BufferedImage(500, 500, BufferedImage.TYPE_INT_RGB) {
          {
            for (int x = 0; x < getWidth(); x++) {
              for (int y = 0; y < getHeight(); y++) {
                final double xf = (x * 1. / getWidth() - .5) * 6;
                final double yf = (y * 1. / getHeight() - .5) * 6;
                final NNResult eval = n.getNetwork().get(0).eval(new NDArray(new int[] { 2 }, new double[] { xf, yf }));
                // int winner = IntStream.range(0, 2).mapToObj(o -> o).max(Comparator.comparing(o -> eval.data.get((int) o))).get();
                final double a = eval.data.get(0);
                final double b = eval.data.get(1);
                // if(Math.random()<0.01) log.debug(String.format("(%s:%s) -> %s vs %s", xf, yf, a, b));
                this.setRGB(x, y, a > b ? 0x1F0000 : 0x001F00);
              }
            }
            final Graphics2D g = (Graphics2D) getGraphics();
            Stream.of(samples).forEach(pt -> {
              final double x = pt[0].get(0);
              final double y = pt[0].get(1);
              final int c = IntStream.range(0, pt[1].dim()).mapToObj(obj -> obj).max(Comparator.comparing(i -> pt[1].get(i))).get();
              g.setColor(Arrays.asList(Color.RED, Color.GREEN).get(c));
              final int xpx = (int) ((x + 3) / 6 * getHeight());
              final int ypx = (int) ((y + 3) / 6 * getHeight());
              g.drawOval(xpx - 1, ypx - 1, 2, 2);
            });
          }
        };
        images.add(img);
      } catch (final Exception e) {
        e.printStackTrace();
      }
      return null;
    });
    try {
      verify(trainer);
    } finally {
      Stream<String> map = images.stream().map(i -> Util.toInlineImage(i, ""));
      String[] array = map.toArray(i -> new String[i]);
      Util.report(array);
    }
  }

  public void verify(final Trainer trainer) {
    trainer.verifyConvergence(0, 0.0, 10);
  }
  
}