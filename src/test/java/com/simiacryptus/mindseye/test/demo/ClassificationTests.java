package com.simiacryptus.mindseye.test.demo;

import java.util.Arrays;
import java.util.function.Function;

import org.junit.Ignore;
import org.junit.Test;

import com.simiacryptus.mindseye.data.GaussianDistribution;
import com.simiacryptus.mindseye.data.Simple2DCircle;
import com.simiacryptus.mindseye.data.Simple2DLine;
import com.simiacryptus.mindseye.data.SnakeDistribution;
import com.simiacryptus.mindseye.data.UnionDistribution;
import com.simiacryptus.mindseye.util.Util;

public abstract class ClassificationTests extends ClassificationTestBase {
  
  public ClassificationTests() {
    super();
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
  public void test_II() throws Exception {
    final double e = 1e-1;
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new Simple2DLine(new double[] { -1, -1 }, new double[] { 1, 1 }),
        new Simple2DLine(new double[] { -1 + e, -1 - e }, new double[] { 1 + e, 1 - e })
        ), 100));
  }

  @Test(expected = RuntimeException.class)
  public void test_III() throws Exception {
    final double e = 1e-1;
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new UnionDistribution(
            new Simple2DLine(new double[] { -1 + e, -1 - e }, new double[] { 1 + e, 1 - e }),
            new Simple2DLine(new double[] { -1 - e, -1 + e }, new double[] { 1 - e, 1 + e })),
            new Simple2DLine(new double[] { -1, -1 }, new double[] { 1, 1 })
        ), 100));
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
  public void test_oo() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new UnionDistribution(new Simple2DCircle(1, new double[] { -0.5, 0 })),
        new UnionDistribution(new Simple2DCircle(1, new double[] { 0.5, 0 }))
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
  @Ignore
  public void test_snakes() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new SnakeDistribution(2, Util.R.get(), 7, 0.01),
        new SnakeDistribution(2, Util.R.get(), 7, 0.01)
        ), 100));
  }

  @Test(expected = RuntimeException.class)
  public void test_sos() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new UnionDistribution(new GaussianDistribution(2, new double[] { 0, 0 }, 0.1)),
        new UnionDistribution(new GaussianDistribution(2, new double[] { -1, 0 }, 0.1), new GaussianDistribution(2, new double[] { 1, 0 }, 0.1))
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
  public void test_xor() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new UnionDistribution(new GaussianDistribution(2, new double[] { 0, 0 }, 0.1), new GaussianDistribution(2, new double[] { 1, 1 }, 0.1)),
        new UnionDistribution(new GaussianDistribution(2, new double[] { 1, 0 }, 0.1), new GaussianDistribution(2, new double[] { 0, 1 }, 0.1))
        ), 100));
  }

  @Test(expected = RuntimeException.class)
  @Ignore
  public void test_O3() throws Exception {
    test(getTrainingData(2, Arrays.<Function<Void, double[]>> asList(
        new UnionDistribution(new GaussianDistribution(2, new double[] { 0, 0 }, 2.)),
        new UnionDistribution(new Simple2DCircle(.25, new double[] { 0, 0 }))
        ), 1000));
  }
  
}