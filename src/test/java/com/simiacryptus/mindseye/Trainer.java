package com.simiacryptus.mindseye;

import groovy.lang.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.esotericsoftware.kryo.Kryo;
import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.test.dev.ImageNetworkDev;
import com.simiacryptus.mindseye.test.dev.TestMNISTDev;

public class Trainer {
  private static final Logger log = LoggerFactory.getLogger(Trainer.class);
  private double mutationAmount = 0.2;
  private int improvementStaleThreshold = 10;
  private double dynamicRate = 0.1;
  private double staticRate = 1.;
  private boolean verbose = false;
  private List<SupervisedTrainingParameters> currentNetworks = new ArrayList<>();
  private Tuple2<List<SupervisedTrainingParameters>, Double> best = null;
  // int timesSinceImprovement = 0;
  private int currentGeneration = 0;
  private int lastImprovementGeneration = 0;
  private int lastMutationGeneration = 0;
  private int lastLocalImprovementGeneration = 0;
  private double rateAdaptionRate = 0.1;
  private double maxDynamicRate = 1.;
  private int minDynamicRate = 0;
  private int loopA = 5;
  private int loopB = 5;
  
  public Trainer() {
  }
  
  public Trainer add(SupervisedTrainingParameters params) {
    this.currentNetworks.add(params);
    return this;
  }
  
  public Trainer add(PipelineNetwork net, NDArray[][] data) {
    return add(new SupervisedTrainingParameters(net, data));
  }
  
  public Double train(final int stopIterations, final double stopError) {
    long startMs = System.currentTimeMillis();
    currentGeneration = 0;
    while (stopIterations > currentGeneration)
    {
      double error = update();
      error = maybeRevertToBest(error);
      if (error < stopError) {
        log.info(String.format("Completed training to %.5f in %.03fs (%s iterations)", error, (System.currentTimeMillis() - startMs) / 1000., currentGeneration));
        return error;
      }
      if (isVerbose()) {
        log.info(String.format("Error: %s", error));
      }
    }
    return null == best ? null : best.getSecond();
  }
  
  private double update() {
    double lastError = Double.NaN;
    double localBest = Double.MAX_VALUE;
    for (int iterationA = 0; iterationA < loopA; iterationA++) {
      double thisError = lastError;
      for (int iterationB = 0; iterationB < loopB; iterationB++) {
        double[] errorArray = trainSet();
        thisError = DoubleStream.of(errorArray).sum();
        updateBest(thisError);
        if (thisError < localBest)
        {
          if (localBest < Double.MAX_VALUE)
          {
            if (isVerbose()) log.debug(String.format("Local Best %s -> %s", localBest, thisError));
            lastLocalImprovementGeneration = currentGeneration;
          }
          localBest = thisError;
        }
        if (verbose)
        {
          log.debug(String.format("Trained Iteration %s Error: %s (%s) with rate %s", currentGeneration, thisError, Arrays.toString(errorArray), dynamicRate));
        }
      }
      if (Double.isFinite(lastError)) {
        updateRate(lastError, thisError);
      }
      lastError = thisError;
    }
    return lastError;
  }
  
  public double maybeRevertToBest(double thisError) {
    if (null != best && timeSinceLocalImprovement() > improvementStaleThreshold) {
      if (best.getSecond() <= thisError) {
        if (isVerbose()) log.debug(String.format("Discarding %s error, best = %s", thisError, best.getSecond()));
        currentNetworks = new Kryo().copy(best.getFirst());
        thisError = best.getSecond();
        lastImprovementGeneration = currentGeneration;
        mutate();
        dynamicRate = Math.abs(dynamicRate);
      }
    }
    return thisError;
  }
  
  public int timeSinceLocalImprovement() {
    return currentGeneration - lastLocalImprovementGeneration;
  }
  
  public int timeSinceImprovement() {
    return currentGeneration - lastImprovementGeneration;
  }
  
  public void updateRate(double lastError, double thisError) {
    double improvement = lastError - thisError;
    double expectedImprovement = lastError * staticRate / 100.;// (50 + totalIterations);
    double idealRate = dynamicRate * expectedImprovement / improvement;
    double prevRate = dynamicRate;
    if (isVerbose()) {
      log.debug(String.format("Ideal Rate: %s (target %s change, actual %s with %s rate)", idealRate, expectedImprovement, improvement, prevRate));
    }
    if (Double.isFinite(idealRate)) {
      dynamicRate += rateAdaptionRate * (Math.max(Math.min(idealRate, maxDynamicRate), minDynamicRate) - dynamicRate);
    }
    if (isVerbose()) log.debug(String.format("Rate %s -> %s", prevRate, dynamicRate));
  }
  
  public int timeSinceMutation() {
    return currentGeneration - lastMutationGeneration;
  }
  
  public void mutate() {
    mutate(getMutationAmount());
  }
  
  public void mutate(double mutationAmount) {
    lastMutationGeneration = currentGeneration;
    if (verbose) log.debug(String.format("Mutating %s by %s", currentNetworks, mutationAmount));
    currentNetworks.stream().forEach(x -> x.getNet().mutate(mutationAmount));
  }
  
  public void updateBest(double error) {
    currentGeneration++;
    if (Double.isFinite(error) && (null == best || best.getSecond() > error)) {
      if (isVerbose()) {
        log.debug(String.format("New best Error %s > %s", error, null == best ? "null" : best.getSecond()));
      }
      best = new Tuple2<List<SupervisedTrainingParameters>, Double>(new Kryo().copy(currentNetworks), error);
      lastImprovementGeneration = currentGeneration;
    }
  }
  
  public void verifyConvergence(final int maxIterations, final double minError, final int trials) {
    final Kryo kryo = new Kryo();
    List<SupervisedTrainingParameters> lastGood = null;
    List<SupervisedTrainingParameters> masterCopy = currentNetworks;
    for (int epoch = 0; epoch < trials; epoch++)
    {
      currentNetworks = kryo.copy(masterCopy);
      currentNetworks.stream().forEach(x -> x.getNet().mutate(1.));
      final double error = train(maxIterations, minError);
      
      if (isVerbose()) {
        log.info("Final Error: {}", error);
      }
      if (!Double.isFinite(error) || error >= minError) {
        log.info(String.format("Failed to converge in trial %s; Best Error=%s: %s", epoch, error, currentNetworks));
        throw new RuntimeException("Failed in trial " + epoch);
      } else {
        lastGood = currentNetworks;
      }
    }
    currentNetworks = lastGood;
  }
  
  public double[] trainSet() {
    double[] error = currentNetworks.stream().mapToDouble(params -> Stream.of(params.getTrainingData()).parallel().mapToDouble(sample -> {
      final NDArray input = sample[0];
      final NDArray output = sample[1];
      final NNResult eval = params.getNet().eval(input);
      final double trialError = eval.errRms(output) * params.getWeight();
      final NDArray delta = eval.delta(dynamicRate * params.getWeight(), output);
      // try {
      // ImageNetworkDev.report(ImageNetworkDev.imageHtml(
      // TestMNISTDev.toImage(new NDArray(new int[]{output.getDims()[0],output.getDims()[1],output.getDims()[2]}, output.data)),
      // TestMNISTDev.toImage(new NDArray(new int[]{eval.data.getDims()[0],eval.data.getDims()[1],eval.data.getDims()[2]}, eval.data.data)),
      // TestMNISTDev.toImage(new NDArray(new int[]{delta.getDims()[0],delta.getDims()[1],delta.getDims()[2]}, delta.data))
      // ));
      // } catch (Exception e) {
      // e.printStackTrace();
      // }
        eval.feedback(delta);
        assert (Double.isFinite(trialError));
        return trialError;
      }).average().getAsDouble()).toArray();
    currentNetworks.stream().forEach(params -> params.getNet().writeDeltas());
    return error;
  }
  
  public double getMutationAmount() {
    return this.mutationAmount;
  }
  
  public Trainer setMutationAmount(final double mutationAmount) {
    this.mutationAmount = mutationAmount;
    return this;
  }
  
  public Trainer setStaticRate(final double rate) {
    this.staticRate = rate;
    return this;
  }
  
  public boolean isVerbose() {
    return this.verbose;
  }
  
  public Trainer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }
  
  public int getImprovementStaleThreshold() {
    return improvementStaleThreshold;
  }
  
  public Trainer setImprovementStaleThreshold(int improvementStaleThreshold) {
    this.improvementStaleThreshold = improvementStaleThreshold;
    return this;
  }
  
  public Tuple2<List<SupervisedTrainingParameters>, Double> getBest() {
    return best;
  }
  
  public double getDynamicRate() {
    return dynamicRate;
  }
  
  public Trainer setDynamicRate(double dynamicRate) {
    this.dynamicRate = dynamicRate;
    return this;
  }
  
  public double getRateAdaptionRate() {
    return rateAdaptionRate;
  }
  
  public Trainer setRateAdaptionRate(double rateAdaptionRate) {
    this.rateAdaptionRate = rateAdaptionRate;
    return this;
  }
  
  public double getMaxDynamicRate() {
    return maxDynamicRate;
  }
  
  public Trainer setMaxDynamicRate(double maxDynamicRate) {
    this.maxDynamicRate = maxDynamicRate;
    return this;
  }
  
  public int getMinDynamicRate() {
    return minDynamicRate;
  }
  
  public Trainer setMinDynamicRate(int minDynamicRate) {
    this.minDynamicRate = minDynamicRate;
    return this;
  }
  
  public int getLoopA() {
    return loopA;
  }
  
  public Trainer setLoopA(int loopA) {
    this.loopA = loopA;
    return this;
  }
  
  public int getLoopB() {
    return loopB;
  }
  
  public Trainer setLoopB(int loopB) {
    this.loopB = loopB;
    return this;
  }
  
}
