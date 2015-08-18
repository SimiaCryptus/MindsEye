package com.simiacryptus.mindseye.test.regression;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
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
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.SigmoidActivationLayer;
import com.simiacryptus.mindseye.layers.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.training.PipelineNetwork;
import com.simiacryptus.mindseye.training.Trainer;

public class MonteCarloClassificationSoftmaxNetworkTests {
  static final Logger log = LoggerFactory.getLogger(MonteCarloClassificationSoftmaxNetworkTests.class);
  
  public static final class Random2DLine implements Function<Void, double[]> {
    
    public final double a;
    public final double b;
    public final double c;
    public final double d;
    
    public Random2DLine() {
      super();
      Random r = Util.R.get();
      this.a = r.nextGaussian();
      this.b = r.nextGaussian();
      this.c = r.nextGaussian();
      this.d = r.nextGaussian();
    }
    
    @Override
    public double[] apply(Void n) {
      double x = Util.R.get().nextDouble() * 4 - 2;
      return new double[] { a * x + b, c * x + d };
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
    final int[] inputSize = new int[] { dimensions };
    final int[] outSize = new int[] { populations.size() };
    final NDArray[][] samples = IntStream.range(0, populations.size()).mapToObj(x -> x)
        .flatMap(p -> IntStream.range(0, 100).mapToObj(i -> {
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
                double xf = (x*1./getWidth() - .5) * 3;
                double yf = (y*1./getHeight() - .5) * 3;
                NNResult eval = n.getNetwork().get(0).eval(new NDArray(new int[]{2},new double[]{xf,yf}));
                int winner = IntStream.range(0, 2).mapToObj(o->o).max(Comparator.comparing(o->eval.data.get((int)o))).get();
                this.setRGB(x, y, 0==winner?0x0F0000:0x000F00);
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
      trainer.verifyConvergence(0, 0.01, 100);
    } finally {
      Util.report((String[])images.stream().map(i->Util.toInlineImage(i, "")).toArray(i->new String[i]));
    }
  }

  public Trainer buildTrainer(final NDArray[][] samples, PipelineNetwork net) {
    Trainer trainer = net.trainer(samples);
    // .setMutationAmplitude(5.)
    // .setVerbose(true)
    // .setStaticRate(.1)
    return trainer;
  }

  public PipelineNetwork buildNetwork() {
    final int[] midSize = new int[] { 10 };
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    PipelineNetwork net = new PipelineNetwork()
        
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize))
//        .add(new BiasLayer(midSize))
//        .add(new SigmoidActivationLayer())
        
        // .add(new DenseSynapseLayer(NDArray.dim(midSize), midSize))
        // .add(new BiasLayer(midSize))
        // .add(new SigmoidActivationLayer())
        
//        .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize))
        .add(new BiasLayer(outSize))

//         .add(new SigmoidActivationLayer());
//        .add(new SoftmaxActivationLayer().setVerbose(false))
        ;
    return net;
  }
  
  @Test
  public void test_Lines() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new Random2DLine(),
        new Random2DLine()
        )));
  }
  
  @Test
  public void test_Gaussians() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new GaussianDistribution(2),
        new GaussianDistribution(2)
        )));
  }
  
  @Test
  public void test_snakes() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new SnakeDistribution(2, Util.R.get(), 7, 0.01),
        new SnakeDistribution(2, Util.R.get(), 7, 0.01)
        )));
  }
  
}
