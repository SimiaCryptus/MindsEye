package com.simiacryptus.mindseye.math;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.polynomials.PolynomialFunction;
import org.apache.commons.math3.analysis.solvers.LaguerreSolver;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.fitting.PolynomialCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoint;
import org.apache.commons.math3.optim.PointValuePair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class UnivariateOptimizer {
  static final Logger log = LoggerFactory.getLogger(UnivariateOptimizer.class);

  public final UnivariateFunction f;
  public double growth = 1.1;
  public double maxValue = 1000;
  public double minValue = 1. / this.maxValue;
  public List<PointValuePair> points = new ArrayList<PointValuePair>();
  public double solveThreshold = -Double.MAX_VALUE;
  private boolean verbose = true;

  public UnivariateOptimizer(final UnivariateFunction f) {
    this.f = f;
  }

  public boolean continueIterating() {
    if (getRelativeUncertianty() < 1e-3) return false;
    if (this.points.get(1).getValue() < this.solveThreshold) return false;
    if (this.points.size() > 10) return false;
    return true;
  }

  public PointValuePair eval(final double optimal) {
    return new PointValuePair(new double[] { optimal }, this.f.value(optimal));
  }

  public double findMin() {
    final PolynomialFunction poly = new PolynomialFunction(PolynomialCurveFitter.create(2).fit(this.points.stream().map(pt -> {
      return new WeightedObservedPoint(1, pt.getFirst()[0], pt.getSecond());
    }).collect(Collectors.toList())));
    final double min = this.points.get(0).getFirst()[0];
    final double max = this.points.get(2).getFirst()[0];
    final double low = this.points.get(1).getFirst()[0];
    try {
      return new LaguerreSolver().solve(1000, poly.polynomialDerivative(),
          min,
          max,
          low);
    } catch (final MathIllegalArgumentException e) {
      return min + (max - min) * Math.random();
    }
  }

  public List<PointValuePair> getKeyPoints() {
    this.points.get(0);
    // PointValuePair left = points.stream().min(Comparator.comparing((PointValuePair x)->x.getFirst()[0])).get();
    final PointValuePair bottom = this.points.stream().min(Comparator.comparing((final PointValuePair x) -> x.getSecond())).get();
    final PointValuePair bottomLeft = this.points.stream().filter(x -> x.getFirst()[0] < bottom.getFirst()[0])
        .max(Comparator.comparing(x -> x.getFirst()[0])).orElseThrow(()->{
          return new RuntimeException();
        });
    final PointValuePair bottomRight = this.points.stream().filter(x -> x.getFirst()[0] > bottom.getFirst()[0])
        .min(Comparator.comparing(x -> x.getFirst()[0])).orElseThrow(()->{
          return new RuntimeException();
        });
    // PointValuePair right = points.stream().max(Comparator.comparing((PointValuePair x)->x.getFirst()[0])).get();
    final List<PointValuePair> newList = Stream.of(new PointValuePair[] {
        bottomLeft,
        bottom,
        bottomRight
    }).distinct().collect(Collectors.toList());
    if (IntStream.range(0, newList.size()).allMatch(i->newList.get(i)==points.get(i))) {
      return this.points;
    } else {
      assert newList.size() == 3;
      return newList;
    }
  }

  public double getRelativeUncertianty() {
    final double a = this.points.get(0).getFirst()[0];
    final double b = this.points.get(1).getFirst()[0];
    final double avg = (b + a) / 2;
    final double span = b - a;
    if (this.isVerbose()) {
      log.debug(String.format("%s span, %s avg; %s conv", span, avg, span / avg));
    }
    if (this.isVerbose()) {
      log.debug(this.points.stream().map(pt -> String.format("%s=%s", Arrays.toString(pt.getFirst()), pt.getSecond()))
          .reduce((aa, bb) -> aa + ", " + bb).get());
    }
    return span / avg;
  }

  public PointValuePair minimize() {
    return minimize(1.);
  }

  public PointValuePair minimize(double start) {
    this.points.add(eval(0));
    this.points.add(eval(start));

    final double oneV = this.points.get(this.points.size() - 1).getValue();
    final double zeroV = this.points.get(this.points.size() - 2).getValue();
    if (oneV > zeroV) {
      for (double x = start / this.growth; true; x /= this.growth) {
        this.points.add(eval(x));
        //Double prevV = this.points.get(this.points.size() - 2).getValue();
        Double lastV = this.points.get(this.points.size() - 1).getValue();
        if (lastV < oneV) {
          break;
        }
        if (x < this.minValue) {
          throw new RuntimeException("x < minValue");
        }
      }
    } else {
      for (double x = start * this.growth; true; x *= this.growth) {
        this.points.add(eval(x));
        Double prevV = this.points.get(this.points.size() - 2).getValue();
        Double thisV = this.points.get(this.points.size() - 1).getValue();
        if (thisV > prevV) {
          break;
        }
        if (x > this.maxValue) {
          throw new RuntimeException("x > maxValue");
        }
      }
    }
    this.points = getKeyPoints();
    while (continueIterating()) {
      this.points.add(eval(findMin()));
      this.points = getKeyPoints();
    }
    return this.points.get(1);
  }

  public boolean isVerbose() {
    return verbose;
  }

  public UnivariateOptimizer setVerbose(boolean verbose) {
    this.verbose = verbose;
    return this;
  }

}