package com.simiacryptus.mindseye.training;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
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

  public synchronized double trainLineSearch() {
    learn(evalTrainingData());
    
    double phi = (Math.sqrt(5)-1)/2;
    
    double errOuterA = Util.geomMean(calcError(evalTrainingData()));
    double posOuterA = 0;
    log.debug(String.format("Evaluating initial position, error %s", errOuterA));
    
    
    double posOuterB = 0.5;
    double currentPos = 0;
    double errOuterB;
    do {
      posOuterB *= 2;
      if(posOuterB > 1000) {
        currentPos = updatePos(currentPos, 0);
        log.debug(String.format("Undefined outer bounds"));
        return Double.POSITIVE_INFINITY;
      }
      currentPos = updatePos(currentPos, posOuterB);
      errOuterB = Util.geomMean(calcError(evalTrainingData()));
      log.debug(String.format("Evaluating initial outer %s, error %s", currentPos, errOuterB));
    } while(errOuterB <= errOuterA);
    
    double windowStopSize = 0.001;
    
    double posInnerA = posOuterB + phi * (posOuterA - posOuterB);
    currentPos = updatePos(currentPos, posInnerA);
    double errInnerA = Util.geomMean(calcError(evalTrainingData()));
    log.debug(String.format("Evaluating initial inner A: %s, error %s", posInnerA, errInnerA));

    double posInnerB = posOuterA + phi * (posOuterB - posOuterA);
    currentPos = updatePos(currentPos, posInnerB);
    double errInnerB = Util.geomMean(calcError(evalTrainingData()));
    log.debug(String.format("Evaluating initial inner B: %s, error %s", posInnerB, errInnerB));

    while(Math.abs(posOuterA - posOuterB) > windowStopSize) {
      if(errInnerA < errInnerB) {
        posOuterB = posInnerB;
        errOuterB = errInnerB;
        
        posInnerB = posOuterB - phi * (posOuterB - posOuterA);
        currentPos = updatePos(currentPos, posInnerB);
        errInnerB = Util.geomMean(calcError(evalTrainingData()));
        log.debug(String.format("Evaluating new inner B: %s, error %s; pos=%s,%s,%s", posInnerB, errInnerB, posOuterA, posInnerA, posOuterB));
      } else {
        posOuterA = posInnerA;
        errOuterA = errInnerA;
        
        posInnerA = posOuterA - phi * (posOuterA - posOuterB);
        currentPos = updatePos(currentPos, posInnerA);
        errInnerA = Util.geomMean(calcError(evalTrainingData()));
        log.debug(String.format("Evaluating new inner A: %s, error %s; pos=%s,%s,%s", posInnerA, errInnerA, posOuterA, posInnerB, posOuterB));
      }
    }

    this.error = calcError(evalTrainingData());
    log.debug(String.format("Terminated at position: %s, error %s", currentPos, this.error));
    return currentPos;
  }

  public double updatePos(double currentPos, double nextPos) {
    double diff = nextPos - currentPos;
    this.currentNetworks.stream().forEach(params -> params.getNet().writeDeltas(diff));
    currentPos += diff;
    return currentPos;
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
    final double geometricMean = Math.exp(DoubleStream.of(error).filter(x -> 0 != x).map(Math::log).average().getAsDouble());
    return Math.pow(geometricMean, 1 / currentNetworks.stream().mapToDouble(p -> p.getWeight()).sum());
  }

}
