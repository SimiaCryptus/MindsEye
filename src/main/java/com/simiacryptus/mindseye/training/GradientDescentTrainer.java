package com.simiacryptus.mindseye.training;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.PointValuePair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.learning.DeltaTransaction;
import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.math.MultivariateOptimizer;

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
    assert (Double.isFinite(dynamicRate));
    this.rate = dynamicRate;
    return this;
  }
  
  public GradientDescentTrainer setVerbose(final boolean verbose) {
    if(verbose) {
      this.verbose = true;
    }
    this.verbose = verbose;
    return this;
  }
  
  public synchronized double[] trainSet() {
    assert(0<currentNetworks.size());
    final List<List<NNResult>> results = evalTrainingData();
    this.error = calcError(results);
    learn(results);
    this.currentNetworks.stream().forEach(params -> params.getNet().writeDeltas(1));
    return this.error;
  }
  
  public synchronized double[] trainLineSearch(int dims) {
    assert(0<currentNetworks.size());
    learn(evalTrainingData());
    double[] one = DoubleStream.generate(() -> 1.).limit(dims).toArray();
    MultivariateFunction f = new MultivariateFunction() {
      double[] pos = new double[dims];
      
      @Override
      public double value(double x[]) {
        double[] diff = new double[x.length];
        for (int i = 0; i < diff.length; i++)
          diff[i] = x[i] - pos[i];
        List<DeltaTransaction> deltaObjs = currentNetworks.stream()
            .flatMap(n -> n.getNet().layers.stream())
            .filter(l -> l instanceof DeltaTransaction)
            .map(l -> (DeltaTransaction) l)
            .distinct().collect(Collectors.toList());
        for (int i = 0; i < diff.length; i++)
          deltaObjs.get(i).write(diff[i]);
        for (int i = 0; i < diff.length; i++)
          pos[i] += diff[i];
        return Util.geomMean(calcError(evalTrainingData()));
      }
    };
    PointValuePair x = new MultivariateOptimizer(f).minimize(one); // May or may not be cloned before evaluations
    f.value(x.getKey()); // f is stateful, based on most recent evaluation
    this.error = calcError(evalTrainingData());
    if (verbose) log.debug(String.format("Terminated at position: %s (%s), error %s", Arrays.toString(x.getKey()), x.getValue(), this.error));
    return x.getKey();
  }
  
  protected void learn(final List<List<NNResult>> results) {
    // Apply corrections
    IntStream.range(0, this.currentNetworks.size()).parallel().forEach(network -> {
      final List<NNResult> netresults = results.get(network);
      final SupervisedTrainingParameters currentNet = this.currentNetworks.get(network);
      IntStream.range(0, netresults.size()).parallel().forEach(sample -> {
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
    if (null == error) return Double.POSITIVE_INFINITY;
    final double geometricMean = Math.exp(DoubleStream.of(error).filter(x -> 0 != x).map(Math::log).average().orElse(Double.POSITIVE_INFINITY));
    return Math.pow(geometricMean, 1 / currentNetworks.stream().mapToDouble(p -> p.getWeight()).sum());
  }
  
  public GradientDescentTrainer copy() {
    return this;
  }
  
  public GradientDescentTrainer clearMomentum() {
    this.currentNetworks.forEach(x -> x.getNet().clearMomentum());
    return this;
  }
  
}
