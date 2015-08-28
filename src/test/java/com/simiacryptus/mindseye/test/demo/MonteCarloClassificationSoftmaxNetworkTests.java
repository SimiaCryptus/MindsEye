package com.simiacryptus.mindseye.test.demo;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.commons.math3.analysis.interpolation.LoessInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.junit.Ignore;
import org.junit.Test;
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

public class MonteCarloClassificationSoftmaxNetworkTests {
  static final Logger log = LoggerFactory.getLogger(MonteCarloClassificationSoftmaxNetworkTests.class);
  
  public static final class Simple2DLine implements Function<Void, double[]> {
    
    public final double width;
    public final double left;
    public final double height;
    public final double bottom;
    
    public Simple2DLine(double[]... pts) {
      super();
      assert (pts.length == 2);
      this.width = (pts[0][0] - pts[1][0]);
      this.left = pts[1][0];
      this.height = (pts[0][1] - pts[1][1]);
      this.bottom = pts[1][1];
    }
    
    public Simple2DLine(double left, double bottom, double width, double height) {
      super();
      this.width = width;
      this.left = left;
      this.height = height;
      this.bottom = bottom;
    }
    
    public Simple2DLine() {
      this(Util.R.get());
    }
    
    public Simple2DLine(Random random) {
      super();
      Random r = random;
      this.width = r.nextGaussian() * 4;
      this.left = r.nextGaussian() - 2;
      this.height = r.nextGaussian() * 4;
      this.bottom = r.nextGaussian() - 2;
    }
    
    @Override
    public double[] apply(Void n) {
      double x = Util.R.get().nextDouble();
      return new double[] { width * x + left, height * x + bottom };
    }
  }
  
  public static final class Simple2DCircle implements Function<Void, double[]> {
    
    private double radius;
    private double[] center;
    
    public Simple2DCircle(double radius, double[] center) {
      super();
      assert (center.length == 2);
      this.radius = radius;
      this.center = center;
    }
    
    @Override
    public double[] apply(Void n) {
      double x = Util.R.get().nextDouble() * 2 * Math.PI;
      return new double[] { Math.sin(x) * radius + center[0], Math.cos(x) * radius + center[1] };
    }
  }
  
  @SuppressWarnings("serial")
  public static final class UnionDistribution extends ArrayList<Function<Void, double[]>> implements Function<Void, double[]> {
    
    public UnionDistribution() {
      super();
    }
    
    @SafeVarargs
    public UnionDistribution(Function<Void, double[]>... c) {
      this(Arrays.asList(c));
    }
    
    public UnionDistribution(Collection<? extends Function<Void, double[]>> c) {
      super(c);
    }
    
    @Override
    public double[] apply(Void t) {
      return get(Util.R.get().nextInt(size())).apply(t);
    }
  }
  
  public static final class GaussianDistribution implements Function<Void, double[]> {
    public final MultivariateNormalDistribution distribution;
    private RealMatrix pos;
    private RealMatrix postMult;
    
    public GaussianDistribution(int dims) {
      Random random = Util.R.get();
      
      final double[] means = new double[dims];
      for (int i = 0; i < means.length; i++)
      {
        means[i] = 1;
      }
      final double[][] diaganals = new double[dims][];
      for (int i = 0; i < diaganals.length; i++)
      {
        diaganals[i] = new double[dims];
        diaganals[i][i] = 1;
      }
      final RandomGenerator rng = new JDKRandomGenerator();
      rng.setSeed(random.nextInt());
      this.distribution = new MultivariateNormalDistribution(rng, means, diaganals);
      
      this.pos = MatrixUtils.createColumnRealMatrix(IntStream.range(0, dims).mapToDouble(i -> random.nextGaussian() * 0.6).toArray());
      this.postMult = MatrixUtils.createRealMatrix(dims, dims);
      IntStream.range(0, dims).forEach(i -> {
        IntStream.range(0, dims).forEach(j -> {
          postMult.setEntry(i, j, random.nextGaussian() * 0.4);
        });
      });
    }
    
    public GaussianDistribution(int dims, double[] pos, double size) {
      Random random = Util.R.get();
      
      final double[] means = new double[dims];
      for (int i = 0; i < means.length; i++)
      {
        means[i] = 1;
      }
      final double[][] diaganals = new double[dims][];
      for (int i = 0; i < diaganals.length; i++)
      {
        diaganals[i] = new double[dims];
        diaganals[i][i] = 1;
      }
      final RandomGenerator rng = new JDKRandomGenerator();
      rng.setSeed(random.nextInt());
      this.distribution = new MultivariateNormalDistribution(rng, means, diaganals);
      
      this.pos = MatrixUtils.createColumnRealMatrix(pos);
      this.postMult = MatrixUtils.createRealMatrix(dims, dims);
      IntStream.range(0, dims).forEach(i -> {
        IntStream.range(0, dims).forEach(j -> {
          postMult.setEntry(i, j, i == j ? size : 0);
        });
      });
    }
    
    @Override
    public double[] apply(Void n) {
      RealMatrix sample = MatrixUtils.createColumnRealMatrix(distribution.sample());
      RealMatrix multiply = this.postMult.multiply(sample);
      RealMatrix add = multiply.add(pos).transpose();
      double[] ds = add.getData()[0];
      return ds;
    }
  }
  
  public final class SnakeDistribution implements Function<Void, double[]>
  {
    public final List<PolynomialSplineFunction> parametricFuctions;
    public final int dims;
    public final double radius;
    public final int nodes;
    
    public SnakeDistribution(int dims, Random random)
    {
      this(dims, random, 10, 0.01);
    }
    
    public SnakeDistribution(int dims, Random random, int nodes)
    {
      this(dims, random, nodes, 0.01);
    }
    
    public SnakeDistribution(int dims, Random random, int nodes, double radius)
    {
      this.dims = dims;
      this.radius = radius;
      this.nodes = nodes;
      final List<PolynomialSplineFunction> parametricFuctions = new ArrayList<PolynomialSplineFunction>();
      for (int j = 0; j < dims; j++)
      {
        final double[] xval = new double[nodes];
        final double[] yval = new double[nodes];
        for (int i = 0; i < xval.length; i++)
        {
          xval[i] = random.nextDouble() * 4 - 2;
          yval[i] = i * 1. / (xval.length - 1);
        }
        parametricFuctions.add(new LoessInterpolator().interpolate(yval, xval));
      }
      
      this.parametricFuctions = parametricFuctions;
    }
    
    @Override
    public double[] apply(Void n) {
      final Random random = Util.R.get();
      final double[] pt = new double[dims];
      final double t = random.nextDouble();
      for (int i = 0; i < pt.length; i++)
      {
        pt[i] = parametricFuctions.get(i).value(t);
        pt[i] += random.nextGaussian() * radius;
      }
      return pt;
    }
    
    public int getDimension()
    {
      return dims;
    }
    
  }
  
  public NDArray[][] getTrainingData(int dimensions, List<Function<Void, double[]>> populations) throws FileNotFoundException, IOException {
    return getTrainingData(dimensions, populations, 100);
  }

  public NDArray[][] getTrainingData(int dimensions, List<Function<Void, double[]>> populations, int sampleN) throws FileNotFoundException, IOException {
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
    PipelineNetwork net = buildNetwork();
    Trainer trainer = buildTrainer(samples, net);
    ArrayList<BufferedImage> images = new ArrayList<>();
    trainer.handler.add(n -> {
      try {
        BufferedImage img = new BufferedImage(500, 500, BufferedImage.TYPE_INT_RGB) {
          {
            for (int x = 0; x < getWidth(); x++) {
              for (int y = 0; y < getHeight(); y++) {
                double xf = (x * 1. / getWidth() - .5) * 6;
                double yf = (y * 1. / getHeight() - .5) * 6;
                NNResult eval = n.getNetwork().get(0).eval(new NDArray(new int[] { 2 }, new double[] { xf, yf }));
                // int winner = IntStream.range(0, 2).mapToObj(o -> o).max(Comparator.comparing(o -> eval.data.get((int) o))).get();
                double a = eval.data.get(0);
                double b = eval.data.get(1);
                //if(Math.random()<0.01) log.debug(String.format("(%s:%s) -> %s vs %s", xf, yf, a, b));
                this.setRGB(x, y, a > b ? 0x1F0000 : 0x001F00);
              }
            }
            Graphics2D g = (Graphics2D) getGraphics();
            Stream.of(samples).forEach(pt -> {
              double x = pt[0].get(0);
              double y = pt[0].get(1);
              int c = IntStream.range(0, pt[1].dim()).mapToObj(obj -> obj).max(Comparator.comparing(i -> pt[1].get(i))).get();
              g.setColor(Arrays.asList(Color.RED, Color.GREEN).get(c));
              int xpx = (int) ((((x + 3) / 6)) * getHeight());
              int ypx = (int) ((((y + 3) / 6)) * getHeight());
              g.drawOval(xpx - 1, ypx - 1, 2, 2);
            });
          }
        };
        images.add(img);
      } catch (Exception e) {
        e.printStackTrace();
      }
      return null;
    });
    try {
      verify(trainer);
    } finally {
      Util.report((String[]) images.stream().map(i -> Util.toInlineImage(i, "")).toArray(i -> new String[i]));
    }
  }
  
  public void verify(Trainer trainer) {
    trainer.verifyConvergence(0, 0.0, 10);
  }
  
  public Trainer buildTrainer(final NDArray[][] samples, PipelineNetwork net) {
    return net.trainer(samples);
    // .setMutationAmplitude(5.)
//    .setVerbose(true);
    // .setStaticRate(.1)
  }
  
  public PipelineNetwork buildNetwork() {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    PipelineNetwork net = new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize))
        .add(new BiasLayer(outSize))
        .add(new SoftmaxActivationLayer());
    return net;
  }
  
  @Test(expected = RuntimeException.class)
  @Ignore
  public void test_Lines() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new Simple2DLine(Util.R.get()),
        new Simple2DLine(Util.R.get())
        ), 100));
  }
  
  @Test(expected = RuntimeException.class)
  public void test_X() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new Simple2DLine(new double[] { -1, -1 }, new double[] { 1, 1 }),
        new Simple2DLine(new double[] { -1, 1 }, new double[] { 1, -1 })
        ), 100));
  }
  
  @Test(expected = RuntimeException.class)
  public void test_II() throws Exception {
    double e = 1e-1;
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new Simple2DLine(new double[] { -1, -1 }, new double[] { 1, 1 }),
        new Simple2DLine(new double[] { -1 + e, -1 - e }, new double[] { 1 + e, 1 - e })
        ), 100));
  }
  
  @Test(expected = RuntimeException.class)
  public void test_III() throws Exception {
    double e = 1e-1;
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new UnionDistribution(
            new Simple2DLine(new double[] { -1 + e, -1 - e }, new double[] { 1 + e, 1 - e }),
            new Simple2DLine(new double[] { -1 - e, -1 + e }, new double[] { 1 - e, 1 + e })),
        new Simple2DLine(new double[] { -1, -1 }, new double[] { 1, 1 })
        ), 100));
  }
  
  @Test(expected = RuntimeException.class)
  @Ignore
  public void test_Gaussians() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new GaussianDistribution(2),
        new GaussianDistribution(2)
        ), 100));
  }
  
  @Test(expected = RuntimeException.class)
  @Ignore
  public void test_snakes() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new SnakeDistribution(2, Util.R.get(), 7, 0.01),
        new SnakeDistribution(2, Util.R.get(), 7, 0.01)
        ), 100));
  }
  
  @Test(expected = RuntimeException.class)
  public void test_xor() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new UnionDistribution(new GaussianDistribution(2, new double[] { 0, 0 }, 0.1), new GaussianDistribution(2, new double[] { 1, 1 }, 0.1)),
        new UnionDistribution(new GaussianDistribution(2, new double[] { 1, 0 }, 0.1), new GaussianDistribution(2, new double[] { 0, 1 }, 0.1))
        ), 100));
  }
  
  @Test(expected = RuntimeException.class)
  public void test_simple() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new UnionDistribution(new GaussianDistribution(2, new double[] { 0, 0 }, 0.1)),
        new UnionDistribution(new GaussianDistribution(2, new double[] { 1, 1 }, 0.1))
        ), 100));
  }
  
  @Test(expected = RuntimeException.class)
  public void test_oo() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new UnionDistribution(new Simple2DCircle(1, new double[] { -0.5, 0 })),
        new UnionDistribution(new Simple2DCircle(1, new double[] { 0.5, 0 }))
        ), 100));
  }

  @Test(expected = RuntimeException.class)
  public void test_O() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new UnionDistribution(new Simple2DCircle(2, new double[] { 0, 0 })),
        new UnionDistribution(new Simple2DCircle(0.1, new double[] { 0, 0 }))
        ), 100));
  }

  @Test(expected = RuntimeException.class)
  public void test_O2() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new UnionDistribution(new Simple2DCircle(2, new double[] { 0, 0 })),
        new UnionDistribution(new Simple2DCircle(1.75, new double[] { 0, 0 }))
        ), 100));
  }

  @Test(expected = RuntimeException.class)
  public void test_O3() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new UnionDistribution(new GaussianDistribution(2, new double[] { 0, 0 }, 2.)),
        new UnionDistribution(new Simple2DCircle(.25, new double[] { 0, 0 }))
        ), 1000));
  }
  
  @Test(expected = RuntimeException.class)
  public void test_sos() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new UnionDistribution(new GaussianDistribution(2, new double[] { 0, 0 }, 0.1)),
        new UnionDistribution(new GaussianDistribution(2, new double[] { -1, 0 }, 0.1), new GaussianDistribution(2, new double[] { 1, 0 }, 0.1))
        ), 100));
  }
  
}
