package com.simiacryptus.mindseye.training;

import java.util.ArrayList;
import java.util.List;
import java.util.function.DoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.univariate.SearchInterval;
import org.apache.commons.math3.optim.univariate.UnivariateObjectiveFunction;
import org.apache.commons.math3.optim.univariate.BrentOptimizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.learning.MassParameters;
import com.simiacryptus.mindseye.learning.NNResult;

public class GradientDescentTrainer {

  private static final Logger log = LoggerFactory.getLogger(GradientDescentTrainer.class);
  
  public List<SupervisedTrainingParameters> currentNetworks = new ArrayList<>();
  private double rate = 0.1;
  private boolean verbose = false;

  double[] error;

  public GradientDescentTrainer() {
  }

  public GradientDescentTrainer add(final PipelineNetwork net, final NDArray[][] data) {
    return add(new SupervisedTrainingParameters(net, data));
  }

  public GradientDescentTrainer add(final SupervisedTrainingParameters params) {
    this.currentNetworks.add(params);
    return this;
  }

  public double getRate() {
    return this.rate;
  }
  
  public boolean isVerbose() {
    return this.verbose;
  }

  public void mutate(final double mutationAmount) {
    if (this.verbose) {
      GradientDescentTrainer.log.debug(String.format("Mutating %s by %s", this.currentNetworks, mutationAmount));
    }
    this.currentNetworks.stream().forEach(x -> x.getNet().mutate(mutationAmount));
  }

  public GradientDescentTrainer setRate(final double dynamicRate) {
    assert(Double.isFinite(dynamicRate));
    this.rate = dynamicRate;
    return this;
  }

  public GradientDescentTrainer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  public synchronized double[] trainSet() {
    final List<List<NNResult>> results = evalTrainingData();
    this.error = calcError(results);
    learn(results);
    this.currentNetworks.stream().forEach(params -> params.getNet().writeDeltas(1));
    return this.error;
  }

  public synchronized double trainLineSearch(double min, double max) {
    learn(evalTrainingData());
    UnivariateFunction f = new UnivariateFunction() {
      
      double pos = 0;
      @Override
      public double value(double x) {
        double diff = x - pos;
        currentNetworks.stream().forEach(params -> params.getNet().writeDeltas(diff));
        pos += diff;
        return Util.geomMean(calcError(evalTrainingData()));
      }
    };
    
    double optimalRate = new BrentOptimizer(1e-8, 1e-10).optimize(new MaxEval(200),
        new UnivariateObjectiveFunction(f),
        GoalType.MINIMIZE,
        new SearchInterval(min, max)).getPoint();
    f.value(optimalRate);
    this.error = calcError(evalTrainingData());
    if(verbose) log.debug(String.format("Terminated at position: %s, error %s", optimalRate, this.error));
    return optimalRate;
  }

  protected void learn(final List<List<NNResult>> results) {
    // Apply corrections
    IntStream.range(0, this.currentNetworks.size()).parallel().forEach(network->{
      final List<NNResult> netresults = results.get(network);
      final SupervisedTrainingParameters currentNet = this.currentNetworks.get(network);
      IntStream.range(0, netresults.size()).parallel().forEach(sample->{
        final NNResult eval = netresults.get(sample);
        final NDArray output = currentNet.getIdeal(eval, currentNet.getTrainingData()[sample][1]);
        final NDArray delta = eval.delta(output).scale(this.rate);
        final double factor = currentNet.getWeight();// * product / rmsList[network];
        if (Double.isFinite(factor)) {
          delta.scale(factor);
        }
        eval.feedback(delta);
      });
    });
  }

  protected double[] calcError(final List<List<NNResult>> results) {
    final List<List<Double>> rms = IntStream.range(0, this.currentNetworks.size()).parallel()
        .mapToObj(network -> {
          final List<NNResult> result = results.get(network);
          final SupervisedTrainingParameters currentNet = this.currentNetworks.get(network);
          return IntStream.range(0, result.size()).parallel().mapToObj(sample -> {
            final NNResult eval = result.get(sample);
            final NDArray output = currentNet.getIdeal(eval, currentNet.getTrainingData()[sample][1]);
            final double err = eval.errRms(output);
            return Math.pow(err, currentNet.getWeight());
          }).collect(Collectors.toList());
        }).collect(Collectors.toList());
    double[] err = rms.stream().map(r ->
        r.stream().mapToDouble(x -> x).filter(Double::isFinite).filter(x -> 0 < x).average().orElse(1))
        .mapToDouble(x -> x).toArray();
    return err;
  }

  protected List<List<NNResult>> evalTrainingData() {
    return this.currentNetworks.parallelStream().map(params -> Stream.of(params.getTrainingData())
        .parallel()
        .map(sample -> {
          final NDArray input = sample[0];
          final NDArray output = sample[1];
          final NNResult eval = params.getNet().eval(input);
          assert eval.data.dim() == output.dim();
          return eval;
        }).collect(Collectors.toList())).collect(Collectors.toList());
  }

  public double[] getError() {
    return error;
  }

  public GradientDescentTrainer setError(double[] error) {
    this.error = error;
    return this;
  }

  public double error() {
    if(null==error) return Double.MAX_VALUE;
    final double geometricMean = Math.exp(DoubleStream.of(error).filter(x -> 0 != x).map(Math::log).average().orElse(Double.MAX_VALUE));
    return Math.pow(geometricMean, 1 / currentNetworks.stream().mapToDouble(p -> p.getWeight()).sum());
  }

  public GradientDescentTrainer copy() {
    return this;
  }

  public GradientDescentTrainer clearMomentum() {
    this.currentNetworks.forEach(x->x.getNet().clearMomentum());
    return this;
  }

}
