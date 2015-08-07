package com.simiacryptus.mindseye.training;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.IntToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.MaxIter;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.SimpleValueChecker;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.PowellOptimizer;
import org.apache.commons.math3.optim.univariate.BrentOptimizer;
import org.apache.commons.math3.optim.univariate.SearchInterval;
import org.apache.commons.math3.optim.univariate.UnivariateObjectiveFunction;
import org.apache.commons.math3.util.DoubleArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.learning.DeltaTransaction;
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

  public synchronized double[] trainLineSearch(int dims) {
    learn(evalTrainingData());
    double[] lowerBounds = new double[dims];
    double[] one = DoubleStream.generate(()->1.).limit(dims).toArray();
    MultivariateFunction f = new MultivariateFunction() {
      double[] pos = new double[dims];
      @Override
      public double value(double x[]) {
        double[] diff = new double[x.length];
        for (int i = 0; i < diff.length; i++) diff[i] = x[i] - pos[i];
        List<DeltaTransaction> deltaObjs = currentNetworks.stream()
            .flatMap(n->n.getNet().layers.stream())
            .filter(l->l instanceof DeltaTransaction)
            .map(l->(DeltaTransaction)l)
            .distinct().collect(Collectors.toList());
        for (int i = 0; i < diff.length; i++) deltaObjs.get(i).write(diff[i]);
        for (int i = 0; i < diff.length; i++) pos[i] += diff[i];
        return Util.geomMean(calcError(evalTrainingData()));
      }
    };
    double f_lower = f.value(lowerBounds);
    double[] upperBounds = IntStream.range(0, dims).mapToDouble(new IntToDoubleFunction() {
      double min = 0;
      double max = 5000;
      @Override
      public double applyAsDouble(int dim) {
        double last = f_lower;
        for(double i=1.;i<max;i*=1.5)
        {
          double[] pt = Arrays.copyOf(lowerBounds, lowerBounds.length);
          pt[dim] = i;
          double x=f.value(pt);
          if(last < x){
            max = i;
            break;
          } else {
            last = x;
          }
        }
        return last;
      }
    }).toArray();
    
    
    final PowellOptimizer optim=new PowellOptimizer(1e-3,1e-3);
    PointValuePair x = optim.optimize(
        GoalType.MINIMIZE,
        new ObjectiveFunction(f),
        new InitialGuess(lowerBounds),
        new MaxEval(1000)//,
        //new SimpleBounds(lowerBounds, upperBounds)
        );
    double optimalRate = x.getValue();
    f.value(x.getKey());
    this.error = calcError(evalTrainingData());
    if(verbose) log.debug(String.format("Terminated at position: %s, error %s", optimalRate, this.error));
    return x.getKey();
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
    if(null==error) return Double.POSITIVE_INFINITY;
    final double geometricMean = Math.exp(DoubleStream.of(error).filter(x -> 0 != x).map(Math::log).average().orElse(Double.POSITIVE_INFINITY));
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
