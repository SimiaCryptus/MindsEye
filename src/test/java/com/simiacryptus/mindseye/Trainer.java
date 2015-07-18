package com.simiacryptus.mindseye;

import groovy.lang.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.esotericsoftware.kryo.Kryo;
import com.simiacryptus.mindseye.learning.NNResult;

public class Trainer {

  private static final Logger log = LoggerFactory.getLogger(Trainer.class);
  
  private Tuple2<List<SupervisedTrainingParameters>, Double> best = null;
  private int currentGeneration = 0;
  private List<SupervisedTrainingParameters> currentNetworks = new ArrayList<>();
  private double dynamicRate = 0.1;
  private int improvementStaleThreshold = 10;
  private double initialMutationAmount = 1.;
  private int lastImprovementGeneration = 0;
  private int lastLocalImprovementGeneration = 0;
  private int lastMutationGeneration = 0;
  private int loopA = 5;
  private int loopB = 1;
  private double maxDynamicRate = 1.;
  private double minDynamicRate = 0;
  private double mutationAmount = 0.2;
  private double rateAdaptionRate = 0.1;
  private double staticRate = 1.;
  private boolean verbose = false;

  public Trainer() {
  }

  public Trainer add(final PipelineNetwork net, final NDArray[][] data) {
    return add(new SupervisedTrainingParameters(net, data));
  }

  public Trainer add(final SupervisedTrainingParameters params) {
    this.currentNetworks.add(params);
    return this;
  }

  public Tuple2<List<SupervisedTrainingParameters>, Double> getBest() {
    return this.best;
  }

  public double getDynamicRate() {
    return this.dynamicRate;
  }

  public int getImprovementStaleThreshold() {
    return this.improvementStaleThreshold;
  }

  public int getLoopA() {
    return this.loopA;
  }

  public int getLoopB() {
    return this.loopB;
  }

  public double getMaxDynamicRate() {
    return this.maxDynamicRate;
  }

  public double getMinDynamicRate() {
    return this.minDynamicRate;
  }

  public double getMutationAmount() {
    return this.mutationAmount;
  }

  public double getRateAdaptionRate() {
    return this.rateAdaptionRate;
  }
  
  public boolean isVerbose() {
    return this.verbose;
  }
  
  public double maybeRevertToBest(double thisError) {
    if (null != this.best && timeSinceLocalImprovement() > this.improvementStaleThreshold) {
      if (this.best.getSecond() <= thisError) {
        if (isVerbose()) {
          Trainer.log.debug(String.format("Discarding %s error, best = %s", thisError, this.best.getSecond()));
          // log.debug(String.format("Discarding %s", best.getFirst().get(0).getNet()));
        }
        this.currentNetworks = Util.kryo().copy(this.best.getFirst());
        thisError = this.best.getSecond();
        this.lastImprovementGeneration = this.currentGeneration;
        mutate();
        this.dynamicRate = Math.abs(this.dynamicRate);
      }
    }
    return thisError;
  }

  public void mutate() {
    mutate(getMutationAmount());
  }

  public void mutate(final double mutationAmount) {
    this.lastMutationGeneration = this.currentGeneration;
    if (this.verbose) {
      Trainer.log.debug(String.format("Mutating %s by %s", this.currentNetworks, mutationAmount));
    }
    this.currentNetworks.stream().forEach(x -> x.getNet().mutate(mutationAmount));
  }

  public Trainer setDynamicRate(final double dynamicRate) {
    this.dynamicRate = dynamicRate;
    return this;
  }

  public Trainer setImprovementStaleThreshold(final int improvementStaleThreshold) {
    this.improvementStaleThreshold = improvementStaleThreshold;
    return this;
  }

  public Trainer setLoopA(final int loopA) {
    this.loopA = loopA;
    return this;
  }

  public Trainer setLoopB(final int loopB) {
    this.loopB = loopB;
    return this;
  }

  public Trainer setMaxDynamicRate(final double maxDynamicRate) {
    this.maxDynamicRate = maxDynamicRate;
    return this;
  }

  public Trainer setMinDynamicRate(final double minDynamicRate) {
    this.minDynamicRate = minDynamicRate;
    return this;
  }

  public Trainer setMutationAmount(final double mutationAmount) {
    this.mutationAmount = mutationAmount;
    return this;
  }

  public Trainer setRateAdaptionRate(final double rateAdaptionRate) {
    this.rateAdaptionRate = rateAdaptionRate;
    return this;
  }

  public Trainer setStaticRate(final double rate) {
    this.staticRate = rate;
    return this;
  }

  public Trainer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  public int timeSinceImprovement() {
    return this.currentGeneration - this.lastImprovementGeneration;
  }

  public int timeSinceLocalImprovement() {
    return this.currentGeneration - this.lastLocalImprovementGeneration;
  }

  public int timeSinceMutation() {
    return this.currentGeneration - this.lastMutationGeneration;
  }

  public Double train(final int stopIterations, final double stopError) {
    final long startMs = System.currentTimeMillis();
    this.currentGeneration = 0;
    while (stopIterations > this.currentGeneration)
    {
      double error = update();
      error = maybeRevertToBest(error);
      if (error < stopError) {
        Trainer.log.info(String.format("Completed training to %.5f in %.03fs (%s iterations)", error, (System.currentTimeMillis() - startMs) / 1000.,
            this.currentGeneration));
        return error;
      }
      if (isVerbose()) {
        Trainer.log.info(String.format("Error: %s", error));
      }
    }
    return null == this.best ? null : this.best.getSecond();
  }

  public double[] trainSet() {
    final List<List<NNResult>> results = this.currentNetworks.parallelStream().map(params -> Stream.of(params.getTrainingData())
        .parallel()
        .map(sample -> {
          final NDArray input = sample[0];
          final NDArray output = sample[1];
          final NNResult eval = params.getNet().eval(input);
          assert eval.data.dim() == output.dim();
          return eval;
        }).collect(Collectors.toList())).collect(Collectors.toList());
    
    final List<List<Double>> rms2 = new ArrayList<>();
    // double totalWeight = 0;
    IntStream.range(0, this.currentNetworks.size()).parallel().forEach(network->{
      final List<NNResult> netresults = results.get(network);
      final SupervisedTrainingParameters currentNet = this.currentNetworks.get(network);
      // totalWeight += currentNet.getWeight();
      final List<Double> rms = new ArrayList<>();
      IntStream.range(0, netresults.size()).parallel().forEach(sample->{
        final NNResult eval = netresults.get(sample);
        final NDArray output = currentNet.getIdeal(eval, currentNet.getTrainingData()[sample][1]);
        final double err = eval.errRms(output);
        synchronized (rms) {
          rms.add(Math.pow(err, currentNet.getWeight()));
        }
      });
      rms2.add(rms);
    });
    final double[] rmsList = rms2.stream().map(r -> r.stream().mapToDouble(x -> x).filter(Double::isFinite).filter(x -> 0 < x).average().orElse(1))
        .mapToDouble(x -> x).toArray();
    // double product = DoubleStream.of(rmsList).filter(x->0!=x).reduce((a,b)->a*b).getAsDouble();
    // double geometricMean = DoubleStream.of(rmsList).filter(x->0!=x).reduce((a,b)->a*b).getAsDouble();
    IntStream.range(0, this.currentNetworks.size()).parallel().forEach(network->{
      final List<NNResult> netresults = results.get(network);
      final SupervisedTrainingParameters currentNet = this.currentNetworks.get(network);
      IntStream.range(0, netresults.size()).parallel().forEach(sample->{
        final NNResult eval = netresults.get(sample);
        final NDArray output = currentNet.getIdeal(eval, currentNet.getTrainingData()[sample][1]);
        final NDArray delta = eval.delta(output).scale(this.dynamicRate);
        final double factor = currentNet.getWeight();// * product / rmsList[network];
        if (Double.isFinite(factor)) {
          delta.scale(factor);
        }
        eval.feedback(delta);
      });
    });
    this.currentNetworks.stream().forEach(params -> params.getNet().writeDeltas());
    return rmsList;
  }

  private double update() {
    double lastError = Double.NaN;
    double localBest = Double.MAX_VALUE;
    for (int iterationA = 0; iterationA < this.loopA; iterationA++) {
      double thisError = lastError;
      for (int iterationB = 0; iterationB < this.loopB; iterationB++) {
        final double[] errorArray = trainSet();
        final double geometricMean = Math.exp(DoubleStream.of(errorArray).filter(x -> 0 != x).map(Math::log).average().getAsDouble());
        thisError = Math.pow(geometricMean, 1 / this.currentNetworks.stream().mapToDouble(p -> p.getWeight()).sum());
        updateBest(thisError);
        if (thisError < localBest)
        {
          if (localBest < Double.MAX_VALUE)
          {
            if (isVerbose()) {
              Trainer.log.debug(String.format("Local Best %s -> %s", localBest, thisError));
            }
            this.lastLocalImprovementGeneration = this.currentGeneration;
          }
          localBest = thisError;
        }
        if (this.verbose)
        {
          Trainer.log.debug(String.format("Trained %s Iteration %s Error: %s (%s) with rate %s",
              Integer.toHexString(System.identityHashCode(this.currentNetworks)), this.currentGeneration, thisError, Arrays.toString(errorArray),
              this.dynamicRate));
        }
      }
      if (Double.isFinite(lastError)) {
        updateRate(lastError, thisError);
      }
      lastError = thisError;
    }
    return lastError;
  }

  public void updateBest(final double error) {
    this.currentGeneration++;
    if (Double.isFinite(error) && (null == this.best || this.best.getSecond() > error)) {
      if (isVerbose()) {
        Trainer.log.debug(String.format("New best Error %s > %s", error, null == this.best ? "null" : this.best.getSecond()));
        // log.debug(String.format("Best: %s", currentNetworks.get(0).getNet()));
      }
      this.best = new Tuple2<List<SupervisedTrainingParameters>, Double>(Util.kryo().copy(this.currentNetworks), error);
      this.lastImprovementGeneration = this.currentGeneration;
    }
  }

  public void updateRate(final double lastError, final double thisError) {
    final double improvement = lastError - thisError;
    final double expectedImprovement = lastError * this.staticRate;// / (50 + currentGeneration);
    final double idealRate = this.dynamicRate * expectedImprovement / improvement;
    final double prevRate = this.dynamicRate;
    if (isVerbose()) {
      Trainer.log.debug(String.format("Ideal Rate: %s (target %s change, actual %s with %s rate)", idealRate, expectedImprovement, improvement, prevRate));
    }
    if (Double.isFinite(idealRate)) {
      this.dynamicRate += this.rateAdaptionRate * (Math.max(Math.min(idealRate, this.maxDynamicRate), this.minDynamicRate) - this.dynamicRate);
    }
    if (isVerbose()) {
      Trainer.log.debug(String.format("Rate %s -> %s", prevRate, this.dynamicRate));
    }
  }

  public void verifyConvergence(final int maxIterations, final double minError, final int trials) {
    final Kryo kryo = Util.kryo();
    List<SupervisedTrainingParameters> lastGood = null;
    final List<SupervisedTrainingParameters> masterCopy = this.currentNetworks;
    for (int epoch = 0; epoch < trials; epoch++)
    {
      this.currentNetworks = kryo.copy(masterCopy);
      this.currentNetworks.stream().forEach(x -> x.getNet().mutate(this.initialMutationAmount));
      final double error = train(maxIterations, minError);

      if (isVerbose()) {
        Trainer.log.info("Final Error: {}", error);
      }
      if (!Double.isFinite(error) || error >= minError) {
        Trainer.log.info(String.format("Failed to converge in trial %s; Best Error=%s: %s", epoch, error, this.currentNetworks));
        throw new RuntimeException("Failed in trial " + epoch);
      } else {
        lastGood = this.currentNetworks;
      }
    }
    this.currentNetworks = lastGood;
    Trainer.log.info("Final result: " + this.currentNetworks.get(0).getNet());
  }

}
