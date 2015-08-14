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
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.PowellOptimizer;
import org.apache.commons.math3.optim.univariate.BrentOptimizer;
import org.apache.commons.math3.optim.univariate.SearchInterval;
import org.apache.commons.math3.optim.univariate.UnivariateObjectiveFunction;
import org.apache.commons.math3.optim.univariate.UnivariatePointValuePair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class UnivariateOptimizer {
  public static final class PtList extends ArrayList<PointValuePair> {

    @Override
    public String toString() {
      StringBuilder builder = new StringBuilder();
      stream().forEach(x->builder.append(String.format("%s%s = %s",builder.length()==0?"":",", Arrays.toString(x.getFirst()),x.getSecond())));
      builder.append("PtList []");
      return builder.toString();
    }
    
  }

  static final Logger log = LoggerFactory.getLogger(UnivariateOptimizer.class);
  
  public final UnivariateFunction f;
  public double growth = 2.;
  public double maxValue = 1e8;
  public double minValue = 0;
  public final List<PointValuePair> points = new PtList();
  double relativeUncertiantyThreshold = 3e-2;
  public double solveThreshold = -Double.MAX_VALUE;
  private boolean verbose = false;

  public UnivariateOptimizer(final UnivariateFunction f) {
    this.f = f;
  }
  
  public boolean continueIterating() {
    if (getRelativeUncertianty() < this.relativeUncertiantyThreshold) return false;
    if (this.points.get(1).getValue() < this.solveThreshold) return false;
    if (this.points.size() > 5) return false;
    return true;
  }
  
  public PointValuePair eval(final double optimal) {
    double value = this.f.value(optimal);
    if(verbose) log.debug(String.format("f(%s) = %s", optimal, value));
    return new PointValuePair(new double[] { optimal }, value);
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
    final PointValuePair bottom = this.points.stream().min(Comparator.comparing((final PointValuePair x) -> x.getSecond())).get();
    final PointValuePair bottomLeft = this.points.stream()
        .filter(x -> x.getFirst()[0] < bottom.getFirst()[0])
        .max(Comparator.comparing(x -> x.getFirst()[0]))
        .orElseThrow(() -> new RuntimeException());
    if(bottom == bottomLeft) {
      return this.points;
    }
    final PointValuePair bottomRight = this.points.stream()
        .filter(x -> x.getFirst()[0] > bottom.getFirst()[0])
        .min(Comparator.comparing(x -> x.getFirst()[0]))
        .orElseThrow(() -> new RuntimeException());
    final List<PointValuePair> newList = Stream.of(new PointValuePair[] {
        bottomLeft,
        bottom,
        bottomRight
    }).distinct().collect(Collectors.toList());
    assert newList.size() == 3;
    if (isVerbose()) {
      UnivariateOptimizer.log.debug(newList.stream()
          .map(pt -> String.format("%s=%s", Arrays.toString(pt.getFirst()), pt.getSecond()))
          .reduce((aa, bb) -> aa + ", " + bb).get());
    }
    return newList;
  }
  
  public double getRelativeUncertianty() {
    final double a = this.points.get(0).getFirst()[0];
    final double b = this.points.get(1).getFirst()[0];
    final double avg = (b + a) / 2;
    final double span = b - a;
    if (isVerbose()) {
      UnivariateOptimizer.log.debug(String.format("%s span, %s avg; %s conv", span, avg, span / avg));
    }
    if (isVerbose()) {
      UnivariateOptimizer.log.debug(this.points.stream()
          .map(pt -> String.format("%s=%s", Arrays.toString(pt.getFirst()), pt.getSecond()))
          .reduce((aa, bb) -> aa + ", " + bb).get());
    }
    return span / avg;
  }
  
  public boolean isVerbose() {
    return this.verbose;
  }
  
  public PointValuePair minimize() {
    return minimize(1.);
  }
  
  public PointValuePair minimize(final double start) {
    this.points.add(eval(0));
    this.points.add(eval(start));
    
    final double oneV = this.points.get(this.points.size() - 1).getValue();
    final double zeroV = this.points.get(this.points.size() - 2).getValue();
    if (oneV > zeroV) {
      for (double x = start / this.growth; x>1e-6; x /= this.growth) {
        this.points.add(eval(x));
        final Double lastV = this.points.get(this.points.size() - 1).getValue();
        if (lastV <= zeroV) {
          break;
        }
        if (x < this.minValue) throw new RuntimeException("x < minValue");
      }
    } else {
      for (double x = start * this.growth; x<maxValue; x *= this.growth) {
        this.points.add(eval(x));
        final Double prevV = this.points.get(this.points.size() - 2).getValue();
        final Double thisV = this.points.get(this.points.size() - 1).getValue();
        if (thisV > prevV) {
          break;
        }
        if (x > this.maxValue) throw new RuntimeException("x > maxValue: " + x);
      }
    }
    try {
      List<PointValuePair> keyPoints = getKeyPoints();
      this.points.clear();
      this.points.addAll(keyPoints);
    } catch (final RuntimeException e) {
      if (this.verbose) {
        UnivariateOptimizer.log.debug("Invalid starting constraints: " + this.points, e);
      }
    }
    assert(3 == this.points.size());
    double leftX = this.points.get(0).getFirst()[0];
    double midX = this.points.get(1).getFirst()[0];
    double rightX = this.points.get(2).getFirst()[0];
    
    final UnivariatePointValuePair optim = new BrentOptimizer(1e-6, 1e-6).optimize(
        GoalType.MINIMIZE,
        new UnivariateObjectiveFunction(f),
        new SearchInterval(leftX,rightX,midX),
        new MaxEval(10000)
        );
    return new PointValuePair(new double[]{optim.getPoint()}, optim.getValue());
  }
  
  public UnivariateOptimizer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }
  
}